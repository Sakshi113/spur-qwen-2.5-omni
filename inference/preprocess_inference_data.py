import os
import json
from tqdm import tqdm

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# 1. List all your raw, messy input JSON and JSONL file paths.
INPUT_DATA_PATHS = [
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/test-seld-starss23-dev-sony.jsonl",
    "/mnt/sandbox/fsaks/spatial_audio/test-QAs/test-spatial-priority-source-intent-starss23-dev-sony.jsonl",
    # Add all other test/inference files here...
]

# 2. Define the directory where the clean files will be saved.
OUTPUT_DIR = "./data/cleaned_inference_data"

# ==============================================================================

def parse_and_clean_entry(entry: dict) -> list[dict]:
    """
    Takes a raw dictionary from the dataset and converts it into a list of
    clean, standardized dictionaries, one for each question.
    Returns an empty list if the entry is invalid.
    """
    cleaned_items = []

    # 1. Validate and get the audio path
    audio_path = entry.get("path") or entry.get("sound")
    if not audio_path or not isinstance(audio_path, str):
        return [] # Skip entry if no valid audio path

    # 2. Validate and get the list of Q&A pairs
    qa_list = entry.get("qa", [])
    if not isinstance(qa_list, list):
        return [] # Skip entry if 'qa' field is not a list

    # 3. Process each Q&A pair defensively
    for qa_item in qa_list:
        # THE FIX for AttributeError: Ensure the item is a dictionary
        if not isinstance(qa_item, dict):
            # This handles cases where an item in the list is just a string
            # or some other malformed data.
            continue

        # Handle nested "qa" keys, which appeared in your data sample
        if "qa" in qa_item and isinstance(qa_item["qa"], dict):
            qa_item = qa_item["qa"]

        question = qa_item.get("question")
        gt_answer = qa_item.get("answer")

        # Ensure both question and answer exist before adding
        if question and gt_answer:
            cleaned_items.append({
                "audio_path": audio_path,
                "question": question,
                "gt_answer": gt_answer,
            })
            
    return cleaned_items


def main():
    """
    Main function to iterate through input files, clean them,
    and save the standardized versions.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting preprocessing. Cleaned files will be saved in: {OUTPUT_DIR}\n")

    for data_path in INPUT_DATA_PATHS:
        if not os.path.exists(data_path):
            print(f"⚠️ Warning: File not found, skipping: {data_path}")
            continue

        print(f"--- Processing file: {os.path.basename(data_path)} ---")
        
        raw_dataset = []
        try:
            with open(data_path, 'r') as f:
                if data_path.endswith('.jsonl'):
                    for line in f:
                        raw_dataset.append(json.loads(line))
                else: # Assume .json
                    raw_dataset = json.load(f)
        except Exception as e:
            print(f"  - ❌ ERROR: Could not parse file {data_path}. Error: {e}. Skipping.")
            continue
        
        all_cleaned_data = []
        for entry in tqdm(raw_dataset, desc="  - Cleaning entries"):
            cleaned_entries = parse_and_clean_entry(entry)
            all_cleaned_data.extend(cleaned_entries)

        # Create the new filename and save the cleaned data
        base_name = os.path.basename(data_path)
        new_filename = f"cleaned_{base_name}"
        output_path = os.path.join(OUTPUT_DIR, new_filename)

        with open(output_path, 'w') as out_file:
            for item in all_cleaned_data:
                out_file.write(json.dumps(item) + '\n')
        
        print(f"  - ✅ Saved {len(all_cleaned_data)} cleaned Q&A pairs to {output_path}")

    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()