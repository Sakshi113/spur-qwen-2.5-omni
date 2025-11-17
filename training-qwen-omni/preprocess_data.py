import os
import json
import random
from tqdm import tqdm

# --- Configuration ---
# 1. List all your input JSON file paths
DATA_ROOT = "/mnt/sandbox/fsaks/spatial_audio"
INPUT_JSON_PATHS = [
    os.path.join(DATA_ROOT, "tasks_QAs/train-af3-arr-2500/dcr-af3.jsonl"),
    os.path.join(DATA_ROOT, "tasks_QAs/train-af3-arr-2500/ean-ss-af3.jsonl"),
    os.path.join(DATA_ROOT, "tasks_QAs/train-af3-arr-2500/gdsc-ss-af3.jsonl"),
    os.path.join(DATA_ROOT, "tasks_QAs/train-af3-arr-2500/pbdm-af3.jsonl"),
    os.path.join(DATA_ROOT, "tasks_QAs/train-af3-arr-2500/scene_reconfig_combined-af3.jsonl"),
    os.path.join(DATA_ROOT, "tasks_QAs/train-af3-arr-2500/spsi-af3.jsonl"),
    os.path.join(DATA_ROOT, "tasks_QAs/train-af3-arr-2500/audio-desc-fine-tune-af3.jsonl"),
]

# 2. Define the output directory and filenames
OUTPUT_DATA_DIR = "./data"
TRAIN_FILENAME = "train_full.jsonl" # Use a new name to be clear
DATASET_INFO_FILENAME = "dataset_info.json"
DATASET_KEY = "spatial_audio_sft_unified_train_only"

# 3. Define the system prompt and special tokens
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving spatial auditory inputs, as well as generating text and speech."
AUDIO_TOKEN = "<audio>"

# --- Main Preprocessing Logic ---
def format_conversation(conversation_list):
    """
    Formats a list of conversation turns into a single string
    following the Qwen chat template, and normalizes keys.
    """
    full_conversation_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    
    for turn in conversation_list:
        role = turn.get("from", turn.get("role", "")).lower()
        text = turn.get("value", turn.get("text", ""))
        
        role = "user" if role == "human" else "assistant" if role == "gpt" else role
        text = text.replace("<sound>", AUDIO_TOKEN)
        
        full_conversation_text += f"<|im_start|>{role}\n{text}<|im_end|>\n"
        
    return full_conversation_text

def main():
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    all_examples = []
    print("Starting data unification process...")

    for file_path in INPUT_JSON_PATHS:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            for entry in tqdm(data, desc=f"Parsing {os.path.basename(file_path)}"):
                audio_path = entry.get("sound", entry.get("audios", [None])[0])

                if not audio_path or not isinstance(audio_path, str):
                    print(f"Skipping entry with invalid audio path: {entry.get('id', 'N/A')}")
                    continue

                formatted_text = format_conversation(entry["conversations"])
                
                all_examples.append({
                    "text": formatted_text,
                    "audio_path": audio_path
                })

    print(f"\nTotal unified examples for training: {len(all_examples)}")
    
    # Shuffle the data before saving
    random.shuffle(all_examples)

    # Write all data to a single training file
    train_path = os.path.join(OUTPUT_DATA_DIR, TRAIN_FILENAME)
    with open(train_path, 'w') as f:
        for item in all_examples:
            f.write(json.dumps(item) + '\n')
    print(f"Successfully wrote all {len(all_examples)} examples to {train_path}")

    # Create the dataset_info.json pointing only to the training file
    dataset_info = {
        DATASET_KEY: {
            "train_file": TRAIN_FILENAME,
        }
    }
    info_path = os.path.join(OUTPUT_DATA_DIR, DATASET_INFO_FILENAME)
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Successfully created {info_path}")

if __name__ == "__main__":
    main()