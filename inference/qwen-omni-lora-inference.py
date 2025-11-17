# import sys
# import os
# import json
# import torch
# import soundfile as sf
# import numpy as np
# import random
# # --- NEW: Import torchaudio for resampling ---
# import torchaudio

# # Ensure your custom transformers code is in the path
# # sys.path.insert(0, '/mnt/sandbox/fsaks/transformers/src')

# from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
# from peft import PeftModel

# # ==============================================================================
# # --- Configuration: Point this to your models and data ---
# # ==============================================================================
# BASE_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
# CHECKPOINT_PATH = "/mnt/sandbox/fsaks/qwen-omni/saves_only_500/qwen2.5omni_spatial_ft_lora_full/checkpoint-500"
# DATA_FILE_PATH = "/mnt/sandbox/fsaks/qwen-omni/data/train_full.jsonl"
# SAMPLE_RATE = 16000 # The target sample rate the model expects
# MAX_AUDIO_SECONDS = 30
# MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # ==============================================================================

# # --- Main Inference Logic ---
# def main():
#     # --- Step 1: Load the Processor and the Fine-Tuned Model ---
#     print("Loading processor...")
#     processor = Qwen2_5OmniProcessor.from_pretrained(BASE_MODEL_ID)
#     # processor.to(DEVICE)

#     print(f"Loading base model '{BASE_MODEL_ID}'...")
#     base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#         BASE_MODEL_ID,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     base_model.disable_talker()

#     print(f"Applying fine-tuned adapter weights from '{CHECKPOINT_PATH}'...")
#     model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
#     model.eval()
#     print("✅ Fine-tuned model loaded successfully!")

#     # --- Step 2: Load and Prepare a Sample from Your Dataset ---
#     print(f"\nLoading data sample from '{DATA_FILE_PATH}'...")
#     with open(DATA_FILE_PATH, 'r') as f:
#         data = json.load(f)
#     sample = random.choice(data)
    
#     user_prompt = ""
#     ground_truth_answer = ""
#     for turn in sample["conversations"]:
#         if turn.get("from") == "human": user_prompt = turn.get("value", "")
#         elif turn.get("from") == "gpt": ground_truth_answer = turn.get("value", "")

#     audio_path = sample.get("sound")
#     if not audio_path or not os.path.exists(audio_path):
#         print(f"❌ ERROR: Audio file not found at path: {audio_path}")
#         return

#     print(f"Loaded audio: {os.path.basename(audio_path)}")

#     # --- Step 3: Preprocess Audio with On-the-Fly Resampling ---
    
#     wav_data, sr = sf.read(audio_path, dtype='float32', always_2d=True)
#     audio_array = wav_data.T # Shape to (C, T)

#     # --- MODIFIED: Resampling Logic ---
#     if sr != SAMPLE_RATE:
#         print(f"⚠️ Audio sample rate is {sr}Hz. Resampling to {SAMPLE_RATE}Hz...")
#         # Convert numpy array to torch tensor for resampling
#         audio_tensor = torch.from_numpy(audio_array)
        
#         # Create the resampler transform
#         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        
#         # Apply the resampling
#         resampled_tensor = resampler(audio_tensor)
        
#         # Convert back to a numpy array for the rest of the pipeline
#         audio_array = resampled_tensor.numpy()
#         print("✅ Resampling complete.")
#     # --- END MODIFICATION ---

#     # Apply the same 30-second truncation as in training
#     if audio_array.shape[1] > MAX_AUDIO_SAMPLES:
#         print(f"Audio is longer than {MAX_AUDIO_SECONDS}s, truncating to the first {MAX_AUDIO_SECONDS}s.")
#         audio_array = audio_array[:, :MAX_AUDIO_SAMPLES]
    
#     audios = [audio_array]

#     # --- Step 4: Format the Final Prompt and Process Inputs ---
#     conversation = [{"role": "user", "content": user_prompt}]
#     text = processor.apply_chat_template(
#         conversation, add_generation_prompt=True, tokenize=False
#     )
#     inputs = processor(text=text, audio=audios, return_tensors="pt").to(DEVICE)

#     # --- Step 5: Generate the Response ---
#     print("\nGenerating response...")
#     with torch.no_grad():
#         generated_ids = model.generate(
#             **inputs, max_new_tokens=512, do_sample=False
#         )
    
#     generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
#     response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

#     # --- Step 6: Display the Results ---
#     print("\n" + "="*50)
#     print("                  INFERENCE RESULTS")
#     print("="*50)
#     print(f"\n[USER PROMPT]:\n{user_prompt.replace('<sound>', '').strip()}")
#     print("-" * 50)
#     print(f"\n[GROUND TRUTH ANSWER]:\n{ground_truth_answer}")
#     print("-" * 50)
#     print(f"\n[MODEL'S GENERATED ANSWER]:\n{response}")
#     print("="*50)

# if __name__ == "__main__":
#     main()

import sys
import os
import json
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
import torchaudio

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel

# ==============================================================================
# --- Configuration ---
# All settings are defined here for easy modification.
# ==============================================================================

# 1. Model and Checkpoint Paths
BASE_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
CHECKPOINT_PATH = "/mnt/sandbox/fsaks/qwen-omni/saves/qwen2.5omni_spatial_ft_final_corrected/checkpoint-3000"

# 2. Input Dataset Paths
#    This script now expects the new format where each JSON object is a single Q&A pair.
INPUT_DATA_PATHS = [
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/final_test_af3_500/dcr-500-af3.json",
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/final_test_af3_500/ean_ss_500-af3.json",
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/final_test_af3_500/gdsc_ss_500-af3.json",
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/final_test_af3_500/pbdm_500-af3.json",
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/final_test_af3_500/scene-config-combined.json",
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/final_test_af3_500/seld_500-af3.json",
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/final_test_af3_500/spsi_500-af3.json",
    # Add any other .json or .jsonl files in the new format here
]

# 3. Output Directory
#    A separate result file for each input dataset will be created here.
OUTPUT_DIR = "./inference_results_arr"

# 4. Audio Preprocessing Constants (MUST MATCH TRAINING)
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE

# 5. Generation Parameters
MAX_NEW_TOKENS = 512

# 6. Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================

def load_model_and_processor(base_model_id, checkpoint_path):
    """Loads the base model, processor, and applies the PEFT adapter."""
    print("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(base_model_id)

    print(f"Loading base model '{base_model_id}'...")
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.disable_talker()

    print(f"Applying fine-tuned adapter weights from '{checkpoint_path}'...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    print("✅ Fine-tuned model loaded successfully!")
    return model, processor

def run_inference(model, processor, audio_path, text_prompt):
    """Runs a single inference pass for a given audio file and text prompt."""
    try:
        wav_data, sr = sf.read(audio_path, dtype='float32', always_2d=True)
    except Exception as e:
        print(f"  - ⚠️  Warning: Could not read audio file {audio_path}. Skipping. Error: {e}")
        return None
        
    audio_array = wav_data.T

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        audio_array = resampler(torch.from_numpy(audio_array)).numpy()

    if audio_array.shape[1] > MAX_AUDIO_SAMPLES:
        audio_array = audio_array[:, :MAX_AUDIO_SAMPLES]
    
    audios = [audio_array]
    
    prompt_with_token = text_prompt.replace("<sound>", processor.audio_token)
    
    conversation = [{"role": "user", "content": prompt_with_token}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    
    inputs = processor(text=text, audio=audios, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    
    generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    return response

def main():
    """
    Main function to iterate through datasets, run batch inference,
    and save results with extended metadata to separate files.
    """
    model, processor = load_model_and_processor(BASE_MODEL_ID, CHECKPOINT_PATH)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nStarting batch inference. Result files will be saved in '{OUTPUT_DIR}'")
    
    for data_path in INPUT_DATA_PATHS:
        if not os.path.exists(data_path):
            print(f"  - ⚠️  Warning: Dataset file not found: {data_path}. Skipping.")
            continue

        output_filename = f"results_{os.path.basename(data_path)}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w') as out_file:
            print(f"\n--- Processing dataset: {os.path.basename(data_path)} ---")
            print(f"--- Results will be saved to: {output_path} ---")

            dataset = []
            try:
                with open(data_path, 'r') as f:
                    if data_path.endswith('.jsonl'):
                        for line in f:
                            dataset.append(json.loads(line))
                    else:
                        dataset = json.load(f)
            except Exception as e:
                print(f"  - ❌ ERROR: Could not parse file {data_path}. Error: {e}. Skipping.")
                continue
            
            for item in tqdm(dataset, desc=f"  - Inferring"):
                # --- MODIFIED: Extract additional metadata fields ---
                question_id = item.get("id", "unknown_id")
                audio_path = item.get("sound")
                caption = item.get("caption", "") # Default to empty string if missing
                dataset_name = item.get("dataset", "")
                task_type = item.get("task-type", "")
                conversations = item.get("conversations", [])
                # --- END MODIFICATION ---

                if not audio_path or not isinstance(conversations, list) or len(conversations) < 2:
                    continue
                
                try:
                    question = conversations[0].get("value")
                    gt_answer = conversations[1].get("value")
                except (IndexError, AttributeError):
                    continue

                if not question or not gt_answer:
                    continue

                prediction = run_inference(model, processor, audio_path, question)

                if prediction is not None:
                    # --- MODIFIED: Construct the new result item with all fields ---
                    result_item = {
                        "id": question_id,
                        "sound": audio_path,
                        "caption": caption,
                        "dataset": dataset_name,
                        "task-type": task_type,
                        "question": question,
                        "gt_answer": gt_answer,
                        "pred": prediction.strip()
                    }
                    # --- END MODIFICATION ---
                    out_file.write(json.dumps(result_item) + '\n')

    print(f"\n✅ Batch inference complete. All result files saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()