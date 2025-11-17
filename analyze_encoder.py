import os
import sys
import torch
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import warnings


from transformers import Qwen2_5OmniConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

# --- FIX #1: Add the path to your custom modules ---
# This ensures Python can find your spatial feature extractor and its dependencies.
# module_path = '/mnt/sandbox/fsaks/transformers/src/transformers/models/qwen2_5_omni'
# if module_path not in sys.path:
#     sys.path.insert(0, module_path)
# --- END FIX #1 ---


# --- Helper: load FOA WAV (expects 4 channels: W, X, Y, Z) ---
def load_foa(path, target_sr=16000):
    audio, sr = sf.read(path, always_2d=True)
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError(f"{path} must be a 4-channel FOA WAV. Got shape {audio.shape}.")
    audio = audio.astype(np.float32, copy=False)
    if sr != target_sr:
        channels = [librosa.resample(y=audio[:, ch], orig_sr=sr, target_sr=target_sr) for ch in range(4)]
        audio = np.stack(channels, axis=1)
    return audio.T, target_sr


# --- Main Analysis Function ---
def analyze_spatial_encoder(model, processor, audio_dict, sampling_rate=16000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # processor.to(device) # Move the processor to the device to run spatial feature extraction on GPU

    embeddings = {}
    raw_spatial_features = {}

    for name, audio_data in audio_dict.items():
        print(f"Processing audio: {name}...")

        inputs = processor(
            text="dummy",
            audio=[audio_data],
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            # --- FIX #2: Slice the padded features to their actual length ---
            # The processor pads features to a max length (e.g., 30000), but the encoder
            # logic is based on the real length from the attention mask.
            actual_feature_len = inputs["feature_attention_mask"].sum().item()
            sliced_input_features = inputs["input_features"][:, :, :actual_feature_len]
            # --- END FIX #2 ---

            encoder_outputs = model(
                # Pass the SLICED and SQUEEZED tensor
                input_features=sliced_input_features.squeeze(0),
                feature_lens=inputs["feature_attention_mask"].sum(-1),
                aftercnn_lens=(inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1,
                spatial_features=inputs.get("spatial_features"),
                output_spatial_features=True,
            )

        embeddings[name] = encoder_outputs.last_hidden_state.mean(dim=0).squeeze().cpu().numpy()
        if hasattr(encoder_outputs, "spatial_features") and encoder_outputs.spatial_features is not None:
            raw_spatial_features[name] = encoder_outputs.spatial_features.squeeze().cpu().numpy()

    # --- Visualization ---
    if raw_spatial_features:
        fig, axs = plt.subplots(1, len(raw_spatial_features), figsize=(15, 4), squeeze=False)
        fig.suptitle("Raw Spatial Features (from SpatialEncoder module)")
        for i, (name, features) in enumerate(raw_spatial_features.items()):
            im = axs[0, i].imshow(features.T, aspect="auto", origin="lower", interpolation="none")
            axs[0, i].set_title(name)
            axs[0, i].set_xlabel("Time Steps")
        axs[0, 0].set_ylabel("Feature Dimension")
        fig.colorbar(im, ax=axs.ravel().tolist())
        plt.tight_layout()
        plt.show()

    # --- Cosine similarity ---
    print("\n--- Cosine Similarity Analysis ---")
    keys = list(embeddings.keys())
    if len(keys) >= 2:
        vec1_name, vec2_name = keys[0], keys[1]
        vec1 = embeddings[vec1_name]
        vec2 = embeddings[vec2_name]
        
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        print(f"Similarity({vec1_name} vs. {vec2_name}):  {cos(vec1, vec2):.4f}")


if __name__ == "__main__":
    # 1) Initialize model and processor
    config = Qwen2_5OmniConfig.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    config.thinker_config.audio_config.spatial_encoder_type = "foa_conv3d"
    config.thinker_config.audio_config.spatial_channels = 4

    audio_encoder = Qwen2_5OmniAudioEncoder(config.thinker_config.audio_config)
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    # 2) Point to your real FOA WAVs
    # foa_paths = {
    #     "left": '/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/14-208-0020_A.wav',
    #     "right": '/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/14-208-0045_A.wav',
    #     "front-right": '/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/1413-121799-0026_A.wav',
    # }
    audio_files = {
        "left": '/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/14-208-0020_A.wav',
        "right": '/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/14-208-0045_A.wav',
        "front-right": '/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/1413-121799-0026_A.wav',
    }

    # The model expects First-Order Ambisonics (FOA), which typically has 4 channels.
    # Let's create a dummy 4-channel input for demonstration.
    # In your real use case, load your actual W, Y, Z, X channels.
    SAMPLING_RATE = 16000

    # --- Initialize Model and Processor ---
    # This only needs to be done once
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    # processor.to("cuda")
    model.to("cuda")

    # --- Loop through each file and process it ---
    for name, path in audio_files.items():
        print(f"\n--- Processing audio: {name} ---")

        # Load the audio, ensuring it's not converted to mono
        multichannel_audio, sr = librosa.load(path, sr=SAMPLING_RATE, mono=False)

        # Sanity check: The model expects 4 channels for FOA
        if multichannel_audio.shape[0] != 4:
            print(f"WARNING: Audio '{name}' has {multichannel_audio.shape[0]} channels, but 4 are expected for FOA spatial features. Skipping.")
            continue

        print(f"DEBUG: Loaded '{name}' with shape: {multichannel_audio.shape}")

        text = f"<|audio_bos|><|AUDIO|><|audio_eos|> what do you hear in the {name} recording?"

        # The 'audios' argument should be a list containing your single multi-channel array
        inputs = processor(text=text, audio=[multichannel_audio], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Check if the processor correctly identified and created the spatial features
        if "spatial_features" in inputs:
            print("✅ SUCCESS (Processor): 'spatial_features' were extracted.")
            print(f"   Shape of spatial_features: {inputs['spatial_features'].shape}")
        else:
            print("❌ FAILURE (Processor): 'spatial_features' were NOT extracted. The issue is likely in the processor logic or input format.")
            continue

        # Run generation to trigger the model's forward pass and check for warnings
        with torch.no_grad():
            # We only need to run a single step to see the debug prints and warnings
            output = model.generate(**inputs, max_new_tokens=5)
        # # 3) Load all FOA files
        # audio_dict = {}
        # for name, p in foa_paths.items():
        #     if not os.path.exists(p):
        #         raise FileNotFoundError(f"Audio file not found: {p}")
        #     foa, sr = load_foa(p, target_sr=16000)
        #     audio_dict[name] = foa

        # # 4) Run the analysis
        # analyze_spatial_encoder(audio_encoder, processor, audio_dict, sampling_rate=16000)
