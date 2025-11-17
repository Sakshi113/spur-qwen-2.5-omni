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
# ==============================================================================
BASE_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
CHECKPOINT_PATH = "/mnt/sandbox/fsaks/qwen-omni/saves/qwen2.5omni_spatial_ft_lora_full/checkpoint-3000"
INPUT_DATA_PATHS = [
    # Add your AV dataset paths here
    "/mnt/sandbox/fsaks/qwen-omni/inference/test-QAs/test-AV-starss23-sev-test-sony.json",
]
OUTPUT_DIR = "./inference_results_av"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Video Processing Settings ---
VIDEO_SAMPLE_RATE_HZ = 1
MAX_TOTAL_VIDEO_FRAMES = 32 # Max frames to sample from the whole video
VIDEO_CHUNK_SIZE = 8       # Process video in chunks of this many frames to avoid OOM
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
    # processor.to(DEVICE)
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

def run_inference_single_pass(model, processor, text_prompt, audio_array=None, video_frames_chunk=None):
    """
    Runs a single forward pass on a chunk of media.
    This is the core function that interacts with the model.
    """
    audios, videos = None, None
    prompt_with_tokens = text_prompt.replace("<sound>", "") # Start clean

    processor_kwargs = {"text": text_prompt, "return_tensors": "pt"}
    
    if audio_array is not None:
        audios = [audio_array]
        prompt_with_tokens += f" {processor.audio_token}"
        processor_kwargs["audio"] = audios

    if video_frames_chunk:
        videos = [video_frames_chunk]
        prompt_with_tokens += f" {processor.video_token}"
        processor_kwargs["videos"] = videos

    conversation = [{"role": "user", "content": prompt_with_tokens.strip()}]
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
    """
    Orchestrates inference for a single data item, handling video chunking.
    """
    question = item["conversations"][0]["value"]
    audio_path = item.get("audio_path") or item.get("sound")
    video_path = item.get("video_path")

    # --- Pre-load all media ---
    audio_array = None
    if audio_path:
        try:
            wav_data, sr = sf.read(audio_path, dtype='float32', always_2d=True)
            audio_array = wav_data.T
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                audio_array = resampler(torch.from_numpy(audio_array)).numpy()
            if audio_array.shape[1] > MAX_AUDIO_SAMPLES:
                audio_array = audio_array[:, :MAX_AUDIO_SAMPLES]
        except Exception:
            audio_array = None # Fail gracefully
            
    all_video_frames = load_video_frames(video_path) if video_path else None

    # --- Perform Inference ---
    if not all_video_frames:
        # Fallback to audio-only inference
        return run_inference_single_pass(model, processor, question, audio_array=audio_array)
    else:
        # Process video in chunks
        chunk_responses = []
        num_chunks = (len(all_video_frames) + VIDEO_CHUNK_SIZE - 1) // VIDEO_CHUNK_SIZE
        
        for i in range(num_chunks):
            start_idx = i * VIDEO_CHUNK_SIZE
            end_idx = start_idx + VIDEO_CHUNK_SIZE
            frame_chunk = all_video_frames[start_idx:end_idx]
            
            chunk_pred = run_inference_single_pass(
                model, processor, question, 
                audio_array=audio_array, 
                video_frames_chunk=frame_chunk
            )
            if chunk_pred:
                chunk_responses.append(f"In segment {i+1}, {chunk_pred}")
        
        # Aggregate the responses from all chunks
        return " ".join(chunk_responses) if chunk_responses else "Could not generate a response for the video."


def main():
    model, processor = load_model_and_processor(BASE_MODEL_ID, CHECKPOINT_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nStarting batch inference. Result files will be saved in '{OUTPUT_DIR}'")
    
    for data_path in INPUT_DATA_PATHS:
        if not os.path.exists(data_path):
            print(f"  - ⚠️  Warning: Dataset file not found: {data_path}. Skipping.")
            continue

        output_filename = f"results_{os.path.basename(data_path)}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'a') as out_file:
            print(f"\n--- Processing dataset: {os.path.basename(data_path)} ---")
            print(f"--- Results will be saved to: {output_path} ---")
            
            dataset = []
            try:
                with open(data_path, 'r') as f:
                    # Handles both .json and .jsonl
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