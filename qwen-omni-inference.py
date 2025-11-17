# import torch, soundfile as sf
# from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
# from qwen_omni_utils import process_mm_info
# import os, io
# import numpy as np

# def _load_path_or_bytes(x):
#     # Accepts str/PathLike, bytes/bytearray, numpy, torch
#     if isinstance(x, (str, os.PathLike)):
#         wav, _ = sf.read(x, always_2d=True)         # (T, C)
#         return wav
#     if isinstance(x, (bytes, bytearray)):
#         wav, _ = sf.read(io.BytesIO(x), always_2d=True)  # (T, C)
#         return wav
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)

# def _to_CxT_float32(arr, idx=0):
#     arr = np.asarray(arr)
#     if arr.ndim == 1:
#         # mono (T,) -> (1, T)
#         arr = arr[None, :]
#     elif arr.ndim == 2:
#         # Ensure channels-first; assume the larger axis is samples
#         # Common case from soundfile: (T, C) with T >> C -> transpose
#         if arr.shape[0] > arr.shape[1]:
#             arr = arr.T  # -> (C, T)
#     else:
#         raise ValueError(f"Audio at index {idx} must be 1D or 2D, got {arr.ndim}D.")
#     return arr.astype(np.float32, copy=False)

# def debug_print_audio(a_list, label="PRE-PROCESSOR"):
#     for i, a in enumerate(a_list):
#         print(f"[{label}] item {i}: shape={np.asarray(a).shape} (C,T)" if np.asarray(a).ndim==2 
#               else f"[{label}] item {i}: shape={np.asarray(a).shape} (expected (C,T))")

# def normalize_audios_from_mm(audios):
#     if audios is None:
#         return None
#     # If your process_mm_info returns dicts, extract the payloads
#     if len(audios) and isinstance(audios[0], dict):
#         # try common keys your util might use
#         keys = ("audio", "path", "array", "data", "bytes")
#         tmp = []
#         for d in audios:
#             for k in keys:
#                 if k in d and d[k] is not None:
#                     tmp.append(d[k])
#                     break
#         audios = tmp
#     if not isinstance(audios, (list, tuple)):
#         audios = [audios]
#     out = []
#     for i, a in enumerate(audios):
#         a = _load_path_or_bytes(a)              # -> numpy (T,C) / (T,) / (C,T)
#         a = _to_CxT_float32(a, idx=i)           # -> (C, T) float32
#         out.append(a)
#     debug_print_audio(out, "PRE-PROCESSOR")
#     return out

# model_id = "Qwen/Qwen2.5-Omni-7B"

# # Load your edited code’s model with auto device map; add flash-attn if you installed it
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     # attn_implementation="flash_attention_2",  # optional
# )
# model.disable_talker()

# processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

# # ADD THIS LINE TO VERIFY THE FILE PATH
# # ======================================================
# import inspect
# print(f"\n>>> PROCESSOR IS BEING LOADED FROM:\n>>> {inspect.getfile(processor.__class__)}\n")
# # ======================================================

# conversation = [
#     {
#         "role": "system",
#         "content": [
#             # If you want **audio output** as well as text, keep this exact wording:
#             {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
#         ],
#     },
#     {
#         "role": "user",
#         "content": [
#             # Use any combination of text / image / audio / video here
#             {"type": "text", "text": "Describe the sound events you hear in the audio."},
#             # {"type": "audio", "audio": "/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/9026-282386-0010_A.wav"},  # or base64/URL via qwen-omni-utils
#             {"type": "audio", "audio": "./dummy_foa_4ch.wav"}
#         ],
#     },
# ]

# # Whether to let the model use audio tracks embedded in videos (if any)
# USE_AUDIO_IN_VIDEO = True

# # Build the chat template text + collect media paths/blobs
# text = processor.apply_chat_template(conversation, add_generation_prompt=True)
# audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# # audios = normalize_audios_from_mm(audios)

# # ADD THIS BLOCK to debug the input to the processor
# # ======================================================
# print("\n--- DEBUGGING INPUTS TO PROCESSOR ---")
# print(f"Type of 'audios' variable: {type(audios)}")
# if isinstance(audios, list):
#     print(f"Number of audio files found: {len(audios)}")
# else:
#     print(f"Value of 'audios' variable: {audios}")
# print("-------------------------------------\n")
# # ======================================================

# bf = processor(
#     text=text,
#     audio=audios,
#     images=images,
#     videos=videos,
#     return_tensors="pt"
# )

# print("input_features:", bf["input_features"].shape)          # [B, Fbank, T_mel]
# print("feature_attention_mask:", bf["feature_attention_mask"].shape)
# if "spatial_features" in bf:
#     print("spatial_features:", bf["spatial_features"].shape)  # [B, C_spat, F_spat, T_spat]

# inputs = bf.to(model.device)


# # Text + audio response:
# text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

# # Decode text
# reply = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
# print("MODEL TEXT:", reply)

# # Save TTS audio if produced (24kHz)
# # if audio is not None:
# #     sf.write("qwen_omni_reply.wav", audio.reshape(-1).float().cpu().numpy(), samplerate=24000)


# qwen-omni-inference.py

# --- FIX #1: Force Python to find your edited code first ---
# This must be at the very top, before importing transformers
import sys
sys.path.insert(0, '/mnt/sandbox/fsaks/transformers/src')
# -------------------------------------------------------------

import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import inspect
import numpy as np

model_id = "Qwen/Qwen2.5-Omni-7B"

# Load your edited code’s model with auto device map
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.disable_talker()

processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

# Verify that the correct processor file is being loaded
print(f"\n>>> PROCESSOR IS BEING LOADED FROM:\n>>> {inspect.getfile(processor.__class__)}\n")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the sound events you hear in the audio."},
            {"type": "audio", "audio": "/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/9026-282386-0010_A.wav"},  # or base64/URL via qwen-omni-utils
        ],
    },
]

# --- FIX #2: Manually process media inputs to avoid mono conversion ---

# 1. Apply chat template to get the text prompt
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

# 2. Manually load the audio file, ensuring it stays multichannel
audio_path = "/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/9026-282386-0010_A.wav"
wav_data, sampling_rate = sf.read(audio_path, dtype='float32', always_2d=True)
# Soundfile loads as (T, C), we need (C, T)
audio_array = wav_data.T
audios = [audio_array] # The processor expects a list of arrays

# 3. For this test, images and videos are empty
images = []
videos = []
# --------------------------------------------------------------------


print("\n--- DEBUGGING INPUTS TO PROCESSOR ---")
print(f"Manually loaded audio shape: {audios[0].shape}")
print("-------------------------------------\n")


# Now, call the processor with the correctly formatted inputs
bf = processor(
    text=text,
    audio=audios,
    return_tensors="pt"
)

print("input_features:", bf["input_features"].shape)
print("feature_attention_mask:", bf["feature_attention_mask"].shape)
if "spatial_features" in bf:
    # This should now exist and have a shape
    print("spatial_features:", bf["spatial_features"].shape)

inputs = bf.to(model.device)

# Whether to let the model use audio tracks embedded in videos (if any)
USE_AUDIO_IN_VIDEO = True

# Generate text response
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

# Decode text
reply = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
print("MODEL TEXT:", reply)