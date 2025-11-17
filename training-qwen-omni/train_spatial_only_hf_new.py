# import os
# import json
# import torch
# import soundfile as sf
# import numpy as np
# import random
# from datasets import load_dataset
# from typing import Dict, Any, List
# from dataclasses import dataclass
# from transformers import TrainingArguments, Trainer
# from collections import defaultdict

# # Import PEFT for LoRA
# from peft import LoraConfig, get_peft_model, TaskType

# # Import your custom model and processor classes
# from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
# from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

# # === Configuration ===
# MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
# DATA_DIR = "./data"
# DATASET_KEY = "spatial_audio_sft_unified_train_only"
# OUTPUT_DIR = "./saves/qwen2.5omni_spatial_ft_final_corrected" # New output dir
# LEARNING_RATE = 2e-4
# EPOCHS = 7
# GRAD_ACCUM = 4
# USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# # --- Audio constants ---
# SAMPLE_RATE = 16000
# MAX_AUDIO_SECONDS = 30
# MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE

# # === Data Collator (Remains the same) ===
# @dataclass
# class SpatialAudioDataCollator:
#     processor: Qwen2_5OmniProcessor
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         texts = [f["text"] for f in features]
#         audio_arrays = []
#         for feature in features:
#             wav, sr = sf.read(feature["audio_path"], dtype="float32", always_2d=True)
#             wav = wav.T
#             if wav.shape[1] > MAX_AUDIO_SAMPLES:
#                 start_idx = random.randint(0, wav.shape[1] - MAX_AUDIO_SAMPLES)
#                 wav = wav[:, start_idx : start_idx + MAX_AUDIO_SAMPLES]
#             elif wav.shape[1] < MAX_AUDIO_SAMPLES:
#                 pad_width = MAX_AUDIO_SAMPLES - wav.shape[1]
#                 wav = np.pad(wav, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
#             audio_arrays.append(wav)
        
#         batch = self.processor(text=texts, audio=audio_arrays, padding=True, return_tensors="pt")
#         batch["labels"] = batch["input_ids"].clone()
#         return batch

# # === Helper to check trainable parameters ===
# def print_trainable_parameters_and_layers(model):
#     """Prints a structured and detailed list of trainable parameters."""
#     trainable_params = 0
#     all_param = 0
#     trainable_modules = defaultdict(lambda: {"params": 0, "count": 0})

#     for name, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#             if "lora" in name:
#                 key = "LLM (LoRA Adapters)"
#             elif "thinker.audio_tower" in name:
#                 key = "Audio Tower (Full FT)"
#             elif "thinker.lm_head" in name:
#                  key = "Language Model Head (Full FT)"
#             else:
#                 key = f"Other Trainable ({'.'.join(name.split('.')[:2])})"
#             trainable_modules[key]["params"] += param.numel()
            
#     print("\n--- Training Strategy Summary ---")
#     for module, info in trainable_modules.items():
#         print(f"  - Component: {module:<30} | Trainable Parameters: {info['params']:,}")
#     print("--------------------------------------------------\n")
#     print(
#         f"TOTAL trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
#     )

# # === NEW: Final, robust function for setting up model parameters ===
# def setup_model_for_training(
#     model: Qwen2_5OmniForConditionalGeneration,
#     tune_whisper_part: bool = False,
#     tune_spatial_part: bool = True
# ):
#     """
#     Configures the Qwen Omni model for training by applying LoRA and then manually unfreezing
#     the layers intended for full fine-tuning. This avoids PEFT wrapper errors.

#     Args:
#         model: The Qwen2.5 Omni model instance.
#         tune_whisper_part (bool): If True, fully fine-tunes the original "Whisper-like" layers.
#         tune_spatial_part (bool): If True, fine-tunes your new spatial processing layers.
#     """
#     print("[INFO] Configuring model for training...")
    
#     # 1. Define LoRA target modules for the main language model.
#     lora_config = LoraConfig(
#         r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
#         # CRITICAL: Do NOT use `modules_to_save` here, as it causes the TypeError.
#     )

#     # 2. Apply PEFT to the top-level model. By default, this freezes all non-LoRA layers.
#     print("[INFO] Applying LoRA adapters to the model...")
#     peft_model = get_peft_model(model, lora_config)
#     print("[INFO] LoRA applied successfully.")

#     # 3. Manually unfreeze all desired layers for full fine-tuning.
#     modules_to_unfreeze = []
    
#     # Always unfreeze the output head to allow gradient flow. This is the fix for the previous RuntimeError.
#     modules_to_unfreeze.append("thinker.lm_head")
    
#     if tune_whisper_part:
#         print("[INFO] Strategy: Unfreezing Whisper-like encoder parts for full FT.")
#         modules_to_unfreeze.extend([
#             "thinker.audio_tower.conv1", 
#             "thinker.audio_tower.conv2", 
#             "thinker.audio_tower.layers"
#         ])

#     if tune_spatial_part:
#         print("[INFO] Strategy: Unfreezing custom spatial parts for full FT.")
#         modules_to_unfreeze.extend([
#             "thinker.audio_tower.spatial_encoder",
#             "thinker.audio_tower.spatial_projection",
#             "thinker.audio_tower.spatial_feature_processor"
#         ])
    
#     print("[INFO] Manually setting requires_grad=True for full fine-tuning layers...")
#     for name, param in peft_model.named_parameters():
#         if any(module_name in name for module_name in modules_to_unfreeze):
#             param.requires_grad = True
            
#     return peft_model

# # === Main Training ===
# def main():
#     processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
#     model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#         MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32
#     )
#     model.config.use_cache = False

#     # --- CHOOSE YOUR FINE-TUNING STRATEGY HERE ---
#     model = setup_model_for_training(
#         model,
#         tune_whisper_part=False,  # Set to True to train the original audio encoder
#         tune_spatial_part=True   # Set to True to train your new spatial layers
#     )
    
#     print("[INFO] Verifying final trainable parameter setup...")
#     print_trainable_parameters_and_layers(model)
    
#     info_path = os.path.join(DATA_DIR, "dataset_info.json")
#     with open(info_path, 'r') as f:
#         dataset_info = json.load(f)[DATASET_KEY]
#     train_dataset = load_dataset("json", data_files=os.path.join(DATA_DIR, dataset_info["train_file"]), split="train")
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
#         save_strategy="steps",
#         save_steps=500,
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

#     print("\n[INFO] Starting training...")
#     trainer.train()
    
#     print("[INFO] Saving final model and processor...")
#     model.save_pretrained(OUTPUT_DIR)
#     processor.save_pretrained(OUTPUT_DIR)

#     print(f"✅ Training complete. Final model and processor saved to {OUTPUT_DIR}")

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
from collections import defaultdict

# Import PEFT for LoRA
from peft import LoraConfig, get_peft_model, TaskType

# Import your custom model and processor classes
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

# === Configuration ===
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DATA_DIR = "./data"
DATASET_KEY = "spatial_audio_sft_unified_train_only"
OUTPUT_DIR = "./saves/qwen2.5omni_spatial_ft_final_corrected" 
LEARNING_RATE = 2e-4
EPOCHS = 7
GRAD_ACCUM = 4
USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# --- Audio constants ---
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE

# === Data Collator (Remains the same) ===
@dataclass
class SpatialAudioDataCollator:
    processor: Qwen2_5OmniProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        audio_arrays = []
        for feature in features:
            wav, sr = sf.read(feature["audio_path"], dtype="float32", always_2d=True)
            wav = wav.T
            if wav.shape[1] > MAX_AUDIO_SAMPLES:
                start_idx = random.randint(0, wav.shape[1] - MAX_AUDIO_SAMPLES)
                wav = wav[:, start_idx : start_idx + MAX_AUDIO_SAMPLES]
            elif wav.shape[1] < MAX_AUDIO_SAMPLES:
                pad_width = MAX_AUDIO_SAMPLES - wav.shape[1]
                wav = np.pad(wav, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            audio_arrays.append(wav)
        
        batch = self.processor(text=texts, audio=audio_arrays, padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch

# === Helper to check trainable parameters (Remains the same) ===
def print_trainable_parameters_and_layers(model):
    trainable_params = 0
    all_param = 0
    trainable_modules = defaultdict(lambda: {"params": 0, "count": 0})
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora" in name:
                key = "LLM (LoRA Adapters)"
            elif "thinker.audio_tower" in name:
                key = "Audio Tower (Full FT)"
            elif "thinker.lm_head" in name:
                 key = "Language Model Head (Full FT)"
            else:
                key = f"Other Trainable ({'.'.join(name.split('.')[:2])})"
            trainable_modules[key]["params"] += param.numel()
    print("\n--- Training Strategy Summary ---")
    for module, info in trainable_modules.items():
        print(f"  - Component: {module:<30} | Trainable Parameters: {info['params']:,}")
    print("--------------------------------------------------\n")
    print(
        f"TOTAL trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

# === CORRECTED function for setting up model parameters ===
def setup_model_for_training(
    model: Qwen2_5OmniForConditionalGeneration,
    tune_whisper_part: bool = False,
    tune_spatial_part: bool = True
):
    """
    Configures the Qwen Omni model for training by applying LoRA and then manually unfreezing
    the layers intended for full fine-tuning. This version uses the correct prefixed names.
    """
    print("[INFO] Configuring model for training...")
    
    # Define LoRA target modules
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Apply PEFT to the top-level model, which freezes all non-LoRA layers by default.
    print("[INFO] Applying LoRA adapters to the model...")
    peft_model = get_peft_model(model, lora_config)
    print("[INFO] LoRA applied successfully.")

    # --- THIS IS THE KEY FIX ---
    # Define the list of modules to unfreeze using their full names *after* PEFT wrapping.
    # The prefix `base_model.model.` is now included.
    modules_to_unfreeze = []
    
    # Always unfreeze the output head
    modules_to_unfreeze.append("base_model.model.thinker.lm_head")
    
    if tune_whisper_part:
        print("[INFO] Strategy: Unfreezing Whisper-like encoder parts for full FT.")
        modules_to_unfreeze.extend([
            "base_model.model.thinker.audio_tower.conv1", 
            "base_model.model.thinker.audio_tower.conv2", 
            "base_model.model.thinker.audio_tower.layers"
        ])

    if tune_spatial_part:
        print("[INFO] Strategy: Unfreezing custom spatial parts for full FT.")
        modules_to_unfreeze.extend([
            "base_model.model.thinker.audio_tower.spatial_encoder",
            "base_model.model.thinker.audio_tower.spatial_projection",
            "base_model.model.thinker.audio_tower.spatial_feature_processor"
        ])
    
    print("[INFO] Manually setting requires_grad=True for full fine-tuning layers...")
    for name, param in peft_model.named_parameters():
        if any(module_name in name for module_name in modules_to_unfreeze):
            param.requires_grad = True
            
    return peft_model

# === Main Training ===
def main():
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32
    )
    model.config.use_cache = False

    # --- CHOOSE YOUR FINE-TUNING STRATEGY HERE ---
    model = setup_model_for_training(
        model,
        tune_whisper_part=True, # As you tested, this should be False
        tune_spatial_part=True   # This should be True to train your new layers
    )
    
    print("[INFO] Verifying final trainable parameter setup...")
    print_trainable_parameters_and_layers(model)
    
    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)[DATASET_KEY]
    train_dataset = load_dataset("json", data_files=os.path.join(DATA_DIR, dataset_info["train_file"]), split="train")
    data_collator = SpatialAudioDataCollator(processor=processor)

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
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n[INFO] Starting training...")
    trainer.train()
    
    print("[INFO] Saving final model and processor...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"✅ Training complete. Final model and processor saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()