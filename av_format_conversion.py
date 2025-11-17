import json
import os
import re

# === INPUT FILES ===
AUDIO_QA_JSON = "/mnt/sandbox/fsaks/spatial_audio/test-QAs/audio_visual_starss23-dev-tau.jsonl"           # input JSONL-like file (each line is a JSON object)
VIDEO_CAPTIONS_JSON = "/mnt/sandbox/fsaks/spatial_audio/captions/starss23_video.json"  # video caption file (list of dicts)
OUTPUT_JSON = "audio_visual_starss23-dev-tau.json"

# === Step 1: Load video caption mapping ===
with open(VIDEO_CAPTIONS_JSON, "r") as vf:
    video_entries = json.load(vf)
video_lookup = {
    os.path.splitext(os.path.basename(v["video_path"]))[0]: v["video_path"]
    for v in video_entries
}

# === Step 2: Parse each line from the input JSON ===
output_data = []

def extract_json_block(error_str):
    """
    Extract the inner JSON block from a malformed Gemini string like:
    'Malformed Gemini response (not JSON):\\njson\\n{ ... }'
    """
    match = re.search(r'\{[\s\S]*\}', error_str)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

with open(AUDIO_QA_JSON, "r") as af:
    for line in af:
        if not line.strip():
            continue
        data = json.loads(line)

        audio_path = data["path"]
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        audio_caption = data.get("audio_caption", "")
        qa_info = data.get("qa", {})

        # Extract embedded JSON containing the actual question and answer
        embedded_json = extract_json_block(qa_info.get("error", ""))
        if not embedded_json:
            continue

        question = embedded_json.get("question", "").strip()
        answer = embedded_json.get("answer", "").strip()
        reasoning = embedded_json.get("reasoning", "")
        task_type = embedded_json.get("task-type", "Spatial AV QA (Multimodal)")
        dataset = embedded_json.get("dataset", "STARSS23")

        # Find corresponding video_path from lookup
        video_path = video_lookup.get(base_name)
        # if not video_path:
        #     # Default to constructed path if not found
        #     video_path = f"/mnt/sandbox/fsaks/spatial_audio/data/STARSS23/video_dev/dev-test-sony/{base_name}.mp4"

        # === Construct output entry ===
        entry = {
            "id": f"{base_name}_qa1",
            "audio_path": audio_path,
            "video_path": video_path,
            "caption": audio_caption,
            "duration": 44.7,  # placeholder
            "conversations": [
                {"from": "human", "value": question + "\n<sound>"},
                {"from": "gpt", "value": answer}
            ],
            "task-type": task_type,
            "dataset": dataset
        }

        output_data.append(entry)

# === Step 3: Write output JSON ===
with open(OUTPUT_JSON, "w") as out_f:
    json.dump(output_data, out_f, indent=2)

print(f"✅ Successfully converted {len(output_data)} QA entries → {OUTPUT_JSON}")
