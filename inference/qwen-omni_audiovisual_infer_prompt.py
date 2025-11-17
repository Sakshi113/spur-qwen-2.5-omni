import sys
import os
import json
import torch
import soundfile as sf
import numpy as np
import torchaudio
from tqdm import tqdm
import cv2  # For video processing

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel

# ==============================================================================
# --- Configuration ---
# All settings are hardcoded here. Modify this section to change behavior.
# ==============================================================================

# 1. Model and Checkpoint Paths
BASE_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
CHECKPOINT_PATH = "/mnt/sandbox/fsaks/qwen-omni/saves/qwen2.5omni_spatial_ft_final_corrected/checkpoint-3000"

# 2. Input Dataset Paths
#    List all the dataset files you want to run inference on.
INPUT_DATA_PATHS = [
    "/mnt/sandbox/fsaks/qwen-omni/inference/test-QAs/audio_visual_starss23-dev-test_video_capt.json",
    # Add any other .json or .jsonl files here
]

# 3. Output Directory
#    A separate result file for each input dataset will be created here.
OUTPUT_DIR = "./"

# 4. System Prompt
#    This instruction is given to the model before every question.
SYSTEM_PROMPT = (
    "You are a highly perceptive audio-visual AI assistant. Your primary task is to analyze the spatial audio, "
    "which is encoded in FOA format and contains the correct directional cues of sounds. "
    "Focus on identifying and reasoning about the spatial location of sounds (e.g., front-left, behind, above) "
    "and their timing within the audio. "
    "Use the video only as a silent reference to help visualize what is happening, but do not rely on it for sound direction. "
    "Always base spatial reasoning and directional answers on the audio first, and refer to the video only for supporting visual details. "
    "Provide clear, direct, and concise answers grounded in the evidence from the media."
)

# 5. Audio and Video Processing Constants
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE
VIDEO_SAMPLE_RATE_HZ = 1
MAX_TOTAL_VIDEO_FRAMES = 32
VIDEO_CHUNK_SIZE = 8       # Process video in chunks of this many frames to avoid OOM

# 6. Generation and Device Settings
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================

def load_video_frames(video_path):
    """Loads up to MAX_TOTAL_VIDEO_FRAMES from a file, sampling at a fixed rate."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        frames = []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // VIDEO_SAMPLE_RATE_HZ)
        frame_count = 0
        while cap.isOpened() and len(frames) < MAX_TOTAL_VIDEO_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        cap.release()
        return frames
    except Exception as e:
        print(f"  - ⚠️  Warning: Failed to process video {video_path}. Error: {e}")
        return None

def load_model_and_processor(base_model_id, checkpoint_path):
    """Loads the base model, processor, and applies the PEFT adapter."""
    print("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(base_model_id)
    print(f"Loading base model '{base_model_id}'...")
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_model.disable_talker()
    print(f"Applying fine-tuned adapter weights from '{checkpoint_path}'...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    print("✅ Fine-tuned model loaded successfully!")
    return model, processor

def run_inference_single_pass(model, processor, conversation, audio_array=None, video_frames_chunk=None):
    """Runs a single forward pass on a chunk of media, using the provided conversation structure."""
    audios, videos = None, None
    processor_kwargs = {"return_tensors": "pt"}
    
    if audio_array is not None:
        audios = [audio_array]
        processor_kwargs["audio"] = audios

    if video_frames_chunk:
        videos = [video_frames_chunk]
        processor_kwargs["videos"] = videos

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    processor_kwargs["text"] = text
    
    inputs = processor(**processor_kwargs).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    
    generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return response

def process_item_with_chunking(model, processor, item):
    """Orchestrates inference for a single data item, handling video chunking."""
    conversations_from_file = item.get("conversations", [])
    audio_path = item.get("audio_path") or item.get("sound")
    video_path = item.get("video_path")
    
    # --- MODIFIED: Build the conversation in the STRICTLY correct format ---
    
    # 1. Start with the system prompt, formatted as a list of dicts
    full_conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    
    # 2. Extract the user question and build the user turn
    user_turn = {"role": "user", "content": []}
    if conversations_from_file and isinstance(conversations_from_file[0], dict):
        question_text = conversations_from_file[0].get("value", "")
        
        # Add the text part of the user's prompt
        user_turn["content"].append({"type": "text", "text": question_text.replace("<sound>", "")})
        
        # Add placeholders for the media that the processor will use
        if audio_path:
            user_turn["content"].append({"type": "audio", "audio": audio_path})
        if video_path:
            user_turn["content"].append({"type": "video", "video": video_path})
            
    full_conversation.append(user_turn)
    # --- END MODIFICATION ---

    audio_array = None
    if audio_path:
        try:
            wav_data, sr = sf.read(audio_path, dtype='float32', always_2d=True)
            audio_array = wav_data.T
            if sr != SAMPLE_RATE:
                audio_array = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(torch.from_numpy(audio_array)).numpy()
            if audio_array.shape[1] > MAX_AUDIO_SAMPLES:
                audio_array = audio_array[:, :MAX_AUDIO_SAMPLES]
        except Exception:
            audio_array = None
            
    all_video_frames = load_video_frames(video_path) if video_path else None

    if not all_video_frames:
        return run_inference_single_pass(model, processor, full_conversation, audio_array=audio_array)
    else:
        chunk_responses = []
        num_chunks = (len(all_video_frames) + VIDEO_CHUNK_SIZE - 1) // VIDEO_CHUNK_SIZE
        for i in range(num_chunks):
            frame_chunk = all_video_frames[i*VIDEO_CHUNK_SIZE : (i+1)*VIDEO_CHUNK_SIZE]
            chunk_pred = run_inference_single_pass(
                model, processor, full_conversation,
                audio_array=audio_array, 
                video_frames_chunk=frame_chunk
            )
            if chunk_pred:
                chunk_responses.append(f"In segment {i+1}, {chunk_pred}")
        return " ".join(chunk_responses) if chunk_responses else "Could not generate a response for the video."

def main():
    print("INSIDE MAIN")
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
                    dataset = [json.loads(line) for line in f] if data_path.endswith('.jsonl') else json.load(f)
            except Exception as e:
                print(f"  - ❌ ERROR: Could not parse file {data_path}. Error: {e}. Skipping.")
                continue
            
            for item in tqdm(dataset, desc=f"  - Inferring"):
                prediction = process_item_with_chunking(model, processor, item)

                if prediction:
                    result_item = {
                        "id": item.get("id", "unknown_id"),
                        "audio_path": item.get("audio_path") or item.get("sound"),
                        "video_path": item.get("video_path"),
                        "caption": item.get("caption", ""),
                        "dataset": item.get("dataset", ""),
                        "task-type": item.get("task-type", ""),
                        "question": item.get("conversations", [{}])[0].get("value"),
                        "gt_answer": item.get("conversations", [{}, {}])[1].get("value"),
                        "pred": prediction.strip()
                    }
                    out_file.write(json.dumps(result_item) + '\n')

    print(f"\n✅ Batch inference complete. All result files saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()