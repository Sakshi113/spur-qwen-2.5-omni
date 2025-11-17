# import os
# import json
# import torch
# import soundfile as sf
# import numpy as np
# from datasets import load_dataset
# from typing import Dict, Any, List
# from dataclasses import dataclass
# from transformers import TrainingArguments, Trainer

# # Import PEFT for LoRA
# from peft import LoraConfig, get_peft_model, TaskType

# from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
# from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration


# # === Configuration ===
# MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
# DATA_DIR = "./data"
# DATASET_KEY = "spatial_audio_sft"
# OUTPUT_DIR = "./saves/qwen2.5omni_spatial_ft_lora" # Changed output dir
# LEARNING_RATE = 2e-4 # LoRA can often use a slightly higher learning rate
# EPOCHS = 3
# GRAD_ACCUM = 4
# USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# # === Dataset Loading Function ===
# def load_and_prepare_dataset(data_dir, dataset_key):
#     """Loads the dataset and adds the full audio path to each example."""
#     info_path = os.path.join(data_dir, "dataset_info.json")
#     with open(info_path, 'r') as f:
#         dataset_info = json.load(f)[dataset_key]
    
#     dataset_path = os.path.join(data_dir, dataset_info["file_name"])
#     raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

#     def resolve_path(example):
#         if "audio" in example:
#             path_key = "audio"
#         elif "audios" in example and isinstance(example["audios"], list) and len(example["audios"]) > 0:
#             path_key = "audios"
#         else:
#             raise ValueError(f"Could not find a valid audio path in example: {example}")

#         if isinstance(example[path_key], list):
#              example["full_audio_path"] = os.path.join(data_dir, example[path_key][0])
#         else:
#              example["full_audio_path"] = os.path.join(data_dir, example[path_key])

#         if not os.path.exists(example["full_audio_path"]):
#             raise FileNotFoundError(f"Audio file not found at: {example['full_audio_path']}")
            
#         example["text"] = "Describe the sound."
#         return example

#     dataset = raw_dataset.map(resolve_path, remove_columns=raw_dataset.column_names)
#     return dataset


# # === Custom Data Collator ===
# @dataclass
# class SpatialAudioDataCollator:
#     processor: Qwen2_5OmniProcessor
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         texts = [f["text"] for f in features]
#         audio_arrays = [sf.read(f["full_audio_path"], dtype="float32", always_2d=True)[0].T for f in features]
        
#         batch = self.processor(
#             text=texts,
#             audio=audio_arrays,
#             padding=True,
#             return_tensors="pt"
#         )
#         batch["labels"] = batch["input_ids"].clone()
#         return batch

# # === Helper to check trainable parameters ===
# def print_trainable_parameters(model):
#     """Prints the number of trainable parameters in the model."""
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
#     )

# # === Main Training ===
# def main():
#     processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
#     model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#         MODEL_ID,
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32
#     )
#     model.config.use_cache = False

#     # --- PEFT & LoRA Configuration ---
    
#     # 1. Freeze all model parameters
#     for param in model.parameters():
#         param.requires_grad = False
        
#     # 2. Define LoRA configuration for the LLM part (thinker.model)
#     lora_config = LoraConfig(
#         r=32,  # Rank of the LoRA matrices. Higher rank = more parameters. 16 or 32 is a good start.
#         lora_alpha=64, # A scaling factor. A common setting is 2*r.
#         lora_dropout=0.05,
#         bias="none",
#         task_type=TaskType.CAUSAL_LM,
#         target_modules=[ # Target the linear layers in the attention and MLP blocks of the LLM
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ]
#     )
    
#     # 3. Apply LoRA to the LLM part only
#     # Note: We are wrapping the top-level model, but PEFT is smart enough
#     # to find the target_modules within the model's hierarchy.
#     model = get_peft_model(model, lora_config)
    
#     # 4. Unfreeze the audio tower for full fine-tuning
#     for name, param in model.named_parameters():
#         if "audio_tower" in name:
#             param.requires_grad = True

#     # 5. Verify the setup by printing trainable parameters
#     print("Trainable parameter setup:")
#     print_trainable_parameters(model)
#     # --- End of PEFT setup ---

#     # if torch.cuda.is_available():
#     #     processor.to(torch.device("cuda"))
#         # The model is moved to GPU by the Trainer automatically

#     train_dataset = load_and_prepare_dataset(DATA_DIR, DATASET_KEY)
#     data_collator = SpatialAudioDataCollator(processor=processor)

#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=GRAD_ACCUM,
#         learning_rate=LEARNING_RATE,
#         num_train_epochs=EPOCHS,
#         remove_unused_columns=False,
#         bf16=USE_BFLOAT16,
#         fp16=False,
#         logging_steps=10,
#         save_steps=100,
#         save_total_limit=2,
#         report_to="none",
#         gradient_checkpointing=True,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=data_collator,
#     )

#     print("\n[INFO] Starting training with LoRA on LLM and full-tuning on Audio Tower...")
#     trainer.train()
    
#     print("[INFO] Saving final model and processor...")
#     trainer.save_model(OUTPUT_DIR)
#     processor.save_pretrained(OUTPUT_DIR)

#     print(f"✅ Training complete. Model and processor saved to {OUTPUT_DIR}")


# if __name__ == "__main__":
#     main()

import os
import json
import torch
import soundfile as sf
import numpy as np
import random
from datasets import load_dataset
from typing import Dict, Any, List
from dataclasses import dataclass
from transformers import TrainingArguments, Trainer

from peft import LoraConfig, get_peft_model, TaskType

from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

# === Configuration ===
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DATA_DIR = "./data"
DATASET_KEY = "spatial_audio_sft_unified_train_only" # Use the new key
OUTPUT_DIR = "./saves/qwen2.5omni_spatial_ft_lora_full_changed_encoder"
LEARNING_RATE = 2e-4
EPOCHS = 7
GRAD_ACCUM = 4
USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# --- NEW: Define audio constants ---
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE

# === Custom Data Collator with Random Cropping ===
@dataclass
class SpatialAudioDataCollator:
    processor: Qwen2_5OmniProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        audio_arrays = []

        # Load audio and apply random cropping
        for feature in features:
            wav, sr = sf.read(feature["audio_path"], dtype="float32", always_2d=True)
            if sr != SAMPLE_RATE:
                # This is a placeholder for resampling if needed, soundfile doesn't do it.
                # You would need a library like librosa for resampling.
                # For now, we assume all your audio is 16kHz.
                pass
            
            wav = wav.T # Shape to (C, T)

            # --- START: Cropping/Padding Logic ---
            if wav.shape[1] > MAX_AUDIO_SAMPLES:
                # Audio is longer than our max, so we randomly crop a 30s segment
                start_idx = random.randint(0, wav.shape[1] - MAX_AUDIO_SAMPLES)
                wav = wav[:, start_idx : start_idx + MAX_AUDIO_SAMPLES]
            elif wav.shape[1] < MAX_AUDIO_SAMPLES:
                # Audio is shorter, so we pad it with zeros to 30s
                # This ensures all feature sequences in a batch have a similar length,
                # which can sometimes improve stability.
                pad_width = MAX_AUDIO_SAMPLES - wav.shape[1]
                wav = np.pad(wav, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            # --- END: Cropping/Padding Logic ---
            
            audio_arrays.append(wav)
        
        batch = self.processor(
            text=texts, audio=audio_arrays, padding=True, return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

# === Helper to check trainable parameters ===
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# === Main Training ===
def main():
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32
    )
    model.config.use_cache = False

    # --- PEFT & LoRA Configuration ---
    for param in model.parameters():
        param.requires_grad = False
    
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = True

    print("Trainable parameter setup:")
    print_trainable_parameters(model)
    
    # if torch.cuda.is_available():
    #     processor.to(torch.device("cuda"))

    # --- Load the Unified Training-Only Dataset ---
    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)[DATASET_KEY]
    
    train_dataset = load_dataset("json", data_files=os.path.join(DATA_DIR, dataset_info["train_file"]), split="train")
    
    data_collator = SpatialAudioDataCollator(processor=processor)

    # --- Update Training Arguments for a Training-Only Run ---
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
        # Save checkpoints based on steps, not evaluation
        save_strategy="steps",
        save_steps=500, # Save a checkpoint every 500 steps
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # No eval_dataset is provided
        data_collator=data_collator,
    )

    print("\n[INFO] Starting training on the full unified dataset...")
    trainer.train()
    
    print("[INFO] Saving final model and processor...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"✅ Training complete. Final model and processor saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()