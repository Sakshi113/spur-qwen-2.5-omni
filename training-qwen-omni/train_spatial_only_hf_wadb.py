#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, List
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, TrainerCallback

# =======================
# Optional: Weights & Biases
# =======================
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# =======================
# PEFT (LoRA)
# =======================
from peft import LoraConfig, get_peft_model, TaskType

# =======================
# Qwen2.5-Omni
# =======================
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration


# =======================
# Config (env-overridable)
# =======================
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
DATASET_KEY = os.environ.get("DATASET_KEY", "spatial_audio_sft_unified_train_only")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./saves/qwen2.5omni_spatial_ft_final_corrected")
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "7"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))

TUNE_WHISPER_PART = os.environ.get("TUNE_WHISPER_PART", "false").lower() in {"1","true","yes","y"}
TUNE_SPATIAL_PART = os.environ.get("TUNE_SPATIAL_PART", "true").lower() in {"1","true","yes","y"}

USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# Audio constants
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "30"))
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE

# W&B run naming
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "spatial-omni")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "qwen2.5omni_spatial_ft_final_corrected")
WANDB_GROUP = os.environ.get("WANDB_GROUP", "qwen2.5omni")
WANDB_LOG_AUDIO_PREVIEWS = int(os.environ.get("WANDB_LOG_AUDIO_PREVIEWS", "4"))  # how many previews to log


# =======================
# Data Collator
# =======================
@dataclass
class SpatialAudioDataCollator:
    processor: Qwen2_5OmniProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        audio_arrays = []
        for feature in features:
            wav, sr = sf.read(feature["audio_path"], dtype="float32", always_2d=True)
            wav = wav.T  # (channels, samples)

            # crop/pad to MAX_AUDIO_SAMPLES
            if wav.shape[1] > MAX_AUDIO_SAMPLES:
                start_idx = random.randint(0, wav.shape[1] - MAX_AUDIO_SAMPLES)
                wav = wav[:, start_idx : start_idx + MAX_AUDIO_SAMPLES]
            elif wav.shape[1] < MAX_AUDIO_SAMPLES:
                pad_width = MAX_AUDIO_SAMPLES - wav.shape[1]
                wav = np.pad(wav, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

            audio_arrays.append(wav)

        batch = self.processor(text=texts, audio=audio_arrays, padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch


# =======================
# Trainable params summary + logging
# =======================
def get_trainable_params_summary(model):
    trainable_params = 0
    all_param = 0
    buckets = defaultdict(int)

    for name, p in model.named_parameters():
        all_param += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            if "lora" in name:
                key = "LLM (LoRA Adapters)"
            elif "thinker.audio_tower" in name:
                key = "Audio Tower (Full FT)"
            elif "thinker.lm_head" in name:
                key = "Language Model Head (Full FT)"
            else:
                key = f"Other Trainable ({'.'.join(name.split('.')[:2])})"
            buckets[key] += p.numel()

    pct = 100.0 * trainable_params / max(1, all_param)
    return all_param, trainable_params, pct, dict(buckets)


def print_and_maybe_log_trainable(model, step=0):
    all_param, trainable_params, pct, buckets = get_trainable_params_summary(model)

    print("\n--- Training Strategy Summary ---")
    for module, n in buckets.items():
        print(f"  - Component: {module:<30} | Trainable Parameters: {n:,}")
    print("--------------------------------------------------")
    print(f"TOTAL trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {pct:.4f}")

    if WANDB_AVAILABLE and wandb.run is not None:
        # Set up explicit x-axis and log at the given step
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        table = wandb.Table(data=[[k, v] for k, v in buckets.items()], columns=["module", "num_params"])
        wandb.log({
            "trainer/global_step": step,
            "train/all_params": all_param,
            "train/trainable_params": trainable_params,
            "train/trainable_pct": pct,
            "train/trainable_param_table": table,
        }, step=step)


# =======================
# LoRA + selective unfreezing
# =======================
def setup_model_for_training(
    model: Qwen2_5OmniForConditionalGeneration,
    tune_whisper_part: bool = False,
    tune_spatial_part: bool = True,
):
    """
    Apply LoRA to LLM blocks, then unfreeze selected audio/spatial modules for full FT.
    Uses fully-qualified param prefixes post-PEFT wrapping.
    """
    print("[INFO] Configuring model for training...")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    print("[INFO] Applying LoRA adapters to the model...")
    peft_model = get_peft_model(model, lora_config)
    print("[INFO] LoRA applied successfully.")

    modules_to_unfreeze = []

    # Always allow head to learn
    modules_to_unfreeze.append("base_model.model.thinker.lm_head")

    if tune_whisper_part:
        print("[INFO] Unfreezing Whisper-like encoder parts for full FT.")
        modules_to_unfreeze.extend([
            "base_model.model.thinker.audio_tower.conv1",
            "base_model.model.thinker.audio_tower.conv2",
            "base_model.model.thinker.audio_tower.layers",
        ])

    if tune_spatial_part:
        print("[INFO] Unfreezing custom spatial parts for full FT.")
        modules_to_unfreeze.extend([
            "base_model.model.thinker.audio_tower.spatial_encoder",
            "base_model.model.thinker.audio_tower.spatial_projection",
            "base_model.model.thinker.audio_tower.spatial_feature_processor",
        ])

    # Flip requires_grad for chosen modules
    print("[INFO] Manually setting requires_grad=True on selected layers...")
    count = 0
    for name, param in peft_model.named_parameters():
        if any(prefix in name for prefix in modules_to_unfreeze):
            if not param.requires_grad:
                param.requires_grad = True
                count += 1
    print(f"[INFO] Unfrozen parameter tensors: {count}")

    return peft_model


# =======================
# Extra W&B callback
# =======================
class ExtraWandbCallback(TrainerCallback):
    def __init__(self, sample_rate: int, preview_count: int = 4):
        self.sample_rate = sample_rate
        self.preview_count = preview_count
        self.logged_previews = False
        self.metrics_defined = False

    def _define_metrics(self):
        if not WANDB_AVAILABLE or wandb.run is None or self.metrics_defined:
            return
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        self.metrics_defined = True

    def on_train_begin(self, args, state, control, **kwargs):
        self._define_metrics()
        if WANDB_AVAILABLE and wandb.run is not None:
            # Ensure step 0 exists for panels
            wandb.log({"trainer/global_step": 0}, step=0)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Mirror HF global_step into our metric so panels always have an x-axis
        if WANDB_AVAILABLE and wandb.run is not None and state.global_step is not None:
            wandb.log({"trainer/global_step": int(state.global_step)}, step=int(state.global_step), commit=False)
        return control

    def maybe_log_audio_previews(self, train_dataset):
        if self.logged_previews or not WANDB_AVAILABLE or wandb.run is None:
            return
        try:
            n = min(self.preview_count, len(train_dataset))
            rows = []
            for i in range(n):
                ex = train_dataset[i]
                wav, sr = sf.read(ex["audio_path"], dtype="float32", always_2d=True)
                mono = wav.T.mean(axis=0)  # simple mono preview
                caption = ex.get("text", "")[:200]
                rows.append([ex["audio_path"], wandb.Audio(mono, sample_rate=sr), caption])
            wandb.log(
                {"preview/train_samples": wandb.Table(data=rows, columns=["path", "audio", "text"])},
                step=0
            )
            self.logged_previews = True
        except Exception as e:
            print(f"[WARN] W&B audio preview logging skipped: {e}")


# =======================
# Main
# =======================
def main():
    # --- W&B init (safe in offline/disabled modes too) ---
    if WANDB_AVAILABLE:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            group=WANDB_GROUP if WANDB_GROUP else None,
            resume=os.environ.get("WANDB_RESUME", None),
            id=os.environ.get("WANDB_RUN_ID", None),
            allow_val_change=True,
        )
        # Store config for provenance
        wandb.config.update({
            "model_id": MODEL_ID,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "grad_accum": GRAD_ACCUM,
            "bf16": USE_BFLOAT16,
            "sample_rate": SAMPLE_RATE,
            "max_audio_seconds": MAX_AUDIO_SECONDS,
            "tune_whisper_part": TUNE_WHISPER_PART,
            "tune_spatial_part": TUNE_SPATIAL_PART,
        }, allow_val_change=True)

    # Load processor & model
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    )
    model.config.use_cache = False

    # Apply LoRA + selective unfreezing
    model = setup_model_for_training(
        model,
        tune_whisper_part=TUNE_WHISPER_PART,
        tune_spatial_part=TUNE_SPATIAL_PART,
    )

    # Trainable summary (console + W&B at step 0)
    print_and_maybe_log_trainable(model, step=0)

    # Load dataset
    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    with open(info_path, "r") as f:
        dataset_info = json.load(f)[DATASET_KEY]

    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(DATA_DIR, dataset_info["train_file"]),
        split="train",
    )

    data_collator = SpatialAudioDataCollator(processor=processor)

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        remove_unused_columns=False,
        bf16=USE_BFLOAT16,
        fp16=False,
        logging_steps=10,
        logging_first_step=True,        # ensures step 0 is logged
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to=["wandb"] if WANDB_AVAILABLE else "none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Add W&B extras: watch + callback + previews
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.watch(model, log="all", log_freq=100, log_graph=False)
        cb = ExtraWandbCallback(sample_rate=SAMPLE_RATE, preview_count=WANDB_LOG_AUDIO_PREVIEWS)
        trainer.add_callback(cb)
        cb.maybe_log_audio_previews(train_dataset)

    print("\n[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving final model and processor...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"status": "completed"})
        wandb.finish()

    print(f"✅ Training complete. Final model and processor saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
