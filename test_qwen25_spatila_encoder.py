#!/usr/bin/env python3
"""
End-to-end tester for the custom FOA spatial encoder integrated into Qwen2.5-Omni.

What this script does:
- Ensures your *local* transformers fork is used (so your modified modeling/config/processor code is picked up).
- Loads Qwen2.5-Omni-7B with a config that enables your spatial encoder (foa_conv3d or foa_hybrid_cnn).
- Moves the custom spatial feature extractor inside the Processor to GPU (processor.to(device)).
- Loads 4‑ch FOA WAVs (W,X,Y,Z) at 16 kHz (resampling if needed).
- Uses the Processor to create both log-mel features and spatial_features (B, C, F, T).
- Registers a forward hook on the audio encoder to capture:
    • last_hidden_state (audio token embeddings)
    • spatial_features (sequence output of your spatial path, post-encoder)
- Runs model.generate() end-to-end to invoke the audio path.
- Prints shapes, generates a short textual output, shows a spatial-feature heatmap (optional), and computes cosine similarities between pooled embeddings across files.

Usage examples:
    python test_qwen25_spatial_encoder.py \
        --transformers-src /mnt/sandbox/fsaks/transformers/src \
        --encoder-type foa_conv3d \
        --device cuda \
        left=/path/to/left.wav right=/path/to/right.wav "front-right=/path/to/front-right.wav"

Notes:
- If you already have your sys.path set to your fork, you can omit --transformers-src.
- The default audio paths are your L3DAS23 samples (change as needed).
- Requires: numpy, torch, librosa, soundfile, matplotlib.
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transformers-src", type=str, default="/mnt/sandbox/fsaks/transformers/src",
                   help="Path to your local transformers repo 'src' directory (so your modified modules are used).")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Omni-7B",
                   help="HF model id to load weights from.")
    p.add_argument("--encoder-type", type=str, default="foa_conv3d", choices=["foa_conv3d", "foa_hybrid_cnn"],
                   help="Spatial encoder type you enabled in your transformers fork.")
    p.add_argument("--spatial-channels", type=int, default=4, help="Number of spatial channels (FOA=4).")
    p.add_argument("--device", type=str, default="cuda", help="Device for model + processor spatial extractor.")
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate for audio.")
    p.add_argument("--plot", action="store_true", help="Show heatmaps of captured spatial features.")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--prompt", type=str, default="<|audio_bos|><|AUDIO|><|audio_eos|> Describe what you hear.")
    p.add_argument("--no-generate", action="store_true", help="Skip text generation; still runs the audio path.")
    p.add_argument("named_wavs", nargs="*", help='Optional name=path items, e.g. left=/path/a.wav right=/path/b.wav')
    return p.parse_args()


# ------------------------------
# Audio utilities
# ------------------------------
def load_foa(path: str, target_sr: int) -> np.ndarray:
    """Load 4‑channel FOA WAV as float32, shape (4, n_samples). Resample if needed."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio not found: {path}")
    audio, sr = sf.read(path, always_2d=True)
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError(f"{path} must be a 4‑channel FOA WAV (W,X,Y,Z). Got shape {audio.shape}.")
    audio = audio.astype(np.float32, copy=False)
    if sr != target_sr:
        chans = [librosa.resample(y=audio[:, ch], orig_sr=sr, target_sr=target_sr) for ch in range(4)]
        audio = np.stack(chans, axis=1)  # (n, 4)
    return audio.T  # (4, n)


def parse_named_wavs(named: List[str]) -> Dict[str, str]:
    mapping = {}
    for item in named:
        if "=" not in item:
            raise ValueError(f"Named WAV must look like name=/path/file.wav, got: {item}")
        k, v = item.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def default_wavs() -> Dict[str, str]:
    # You can edit these to point to your own local files.
    return {
        "left": "/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/14-208-0020_A.wav",
        "right": "/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/14-208-0045_A.wav",
        "front-right": "/mnt/sandbox/fsaks/spatial_audio/data/L3DAS23_data/Task1/L3DAS23_Task1_train360/data/1413-121799-0026_A.wav",
    }


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()

    # 1) Ensure local transformers fork is used FIRST on sys.path
    if args.transformers_src and os.path.isdir(args.transformers_src):
        if args.transformers_src not in sys.path:
            sys.path.insert(0, args.transformers_src)

    # Sanity print where 'transformers' is imported from
    import transformers, inspect
    print(f"[INFO] Using transformers from: {inspect.getfile(transformers)}")

    # 2) Import *after* sys.path is set so we pick up your modified modules
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration
    from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[INFO] Device: {device}")

    # 3) Build config that enables your spatial encoder
    config = Qwen2_5OmniConfig.from_pretrained(args.model_id)
    # Enable your custom audio encoder path
    config.thinker_config.audio_config.spatial_encoder_type = args.encoder_type
    config.thinker_config.audio_config.spatial_channels = args.spatial_channels

    # 4) Load model + processor and move to device
    print("[INFO] Loading model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_id, config=config)
    model.to(device)
    model.eval()

    print("[INFO] Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    # IMPORTANT: move the processor's internal spatial extractor to the device
    if hasattr(processor, "to"):
        # processor.to(device)
        processor.to("cuda")

    
    # if hasattr(processor, "spatial_feature_extractor") and processor.spatial_feature_extractor is not None:
    #     processor.spatial_feature_extractor.to("cuda")
    # setattr(processor, "_device", torch.device("cuda"))

    # 5) Prepare audio set
    file_map = default_wavs()
    if args.named_wavs:
        file_map.update(parse_named_wavs(args.named_wavs))

    # Validate and load audio
    name_to_audio = {}
    for name, path in file_map.items():
        try:
            foa = load_foa(path, args.sr)
        except Exception as e:
            print(f"[WARN] Skipping '{name}' ({path}): {e}")
            continue
        print(f"[OK] Loaded '{name}' from {path}, shape={foa.shape}, sr={args.sr}")
        if foa.shape[0] != args.spatial_channels:
            print(f"[WARN] '{name}' has {foa.shape[0]} channels but --spatial-channels is {args.spatial_channels}.")
        name_to_audio[name] = foa

    if not name_to_audio:
        raise SystemExit("No valid audio loaded. Please provide at least one 4‑ch FOA WAV.")

    # 6) Forward hook to capture audio encoder outputs
    captured = {
        "last_hidden_state": {},
        "spatial_features": {},
    }

    def audio_encoder_hook(module, inputs, output):
        # output is Qwen2_5OmniAudioEncoderOutput in your fork
        try:
            if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                captured["last_hidden_state"][current_name] = output.last_hidden_state.detach().float().cpu()
            if hasattr(output, "spatial_features") and output.spatial_features is not None:
                captured["spatial_features"][current_name] = output.spatial_features.detach().float().cpu()
        except Exception as e:
            print(f"[HOOK WARN] Could not capture outputs: {e}")

    enc = getattr(model, "thinker", None)
    if enc is None or not hasattr(enc, "audio_tower"):
        raise RuntimeError("Could not locate audio encoder at model.thinker.audio_tower")
    hook_handle = enc.audio_tower.register_forward_hook(audio_encoder_hook)

    # 7) Loop through files and run end-to-end generation to trigger the audio path
    pooled = {}
    for current_name, foa in name_to_audio.items():
        print(f"\n[RUN] Processing: {current_name}")
        text = f"<|audio_bos|><|AUDIO|><|audio_eos|> What do you hear in the {current_name} recording?"
        # Build batch = 1
        inputs = processor(text=text, audio=[foa], return_tensors="pt")
        # Move tensors to the same device as the model
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # Basic sanity about spatial_features presence
        if "spatial_features" in inputs:
            print(f"[OK] spatial_features present: shape={tuple(inputs['spatial_features'].shape)}")
        else:
            print("[WARN] spatial_features missing in Processor output. Check your processing_qwen2_5_omni.py.")
        
        with torch.no_grad():
            if not args.no_generate:
                gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=(model.generation_config.pad_token_id or model.generation_config.eos_token_id),
                        return_dict_in_generate=True,  # <— ensures we can read .sequences
                    )                # Decode only the newly generated tokens
                # out = processor.tokenizer.batch_decode(gen_ids[:, -args.max_new_tokens:], skip_special_tokens=True)
                # print(f"[GEN] {current_name}: {out[0]!r}")
                # Accept both output styles
        import torch as _torch
        if isinstance(gen, _torch.Tensor):
            sequences = gen
        elif hasattr(gen, "sequences"):
            sequences = gen.sequences
        elif isinstance(gen, (tuple, list)) and len(gen) > 0 and isinstance(gen[0], _torch.Tensor):
            sequences = gen[0]
        else:
            raise TypeError(f"Unexpected generate() return type: {type(gen)}")

        # Decode only the newly generated tokens
        out = processor.tokenizer.batch_decode(
            sequences[:, -args.max_new_tokens:],
            skip_special_tokens=True
        )
        print(f"[GEN] {current_name}: {out[0]!r}")

        # Pool last_hidden_state if we captured it
        if current_name in captured["last_hidden_state"]:
            emb = captured["last_hidden_state"][current_name]
            # Mean pool over time (dim=1) -> (seq, dim) or (batch, seq, dim)? handle both
            if emb.dim() == 3:  # (B, T, D)
                pooled[current_name] = emb.mean(dim=1).squeeze(0).numpy()
            elif emb.dim() == 2:  # (T, D)
                pooled[current_name] = emb.mean(dim=0).numpy()
            else:
                print(f"[WARN] Unexpected embedding shape for {current_name}: {emb.shape}")
        else:
            print(f"[WARN] No last_hidden_state captured for {current_name}.")

    # Remove hook
    hook_handle.remove()

    # 8) Optional: visualize captured spatial features
    if args.plot and captured["spatial_features"]:
        names = list(captured["spatial_features"].keys())
        n = len(names)
        import math as _math
        cols = min(3, n)
        rows = int(_math.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
        fig.suptitle("Captured Spatial Encoder Sequence (post-encoder)")
        for idx, nm in enumerate(names):
            r, c = divmod(idx, cols)
            feat = captured["spatial_features"][nm]  # shape (B?, T, D) or similar depending on your code
            arr = feat.squeeze(0).detach().numpy()
            # If it looks like (T, D), transpose for display
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            im = axs[r, c].imshow(arr, aspect="auto", origin="lower", interpolation="nearest")
            axs[r, c].set_title(nm)
            axs[r, c].set_xlabel("Time")
            axs[r, c].set_ylabel("Feature")
            fig.colorbar(im, ax=axs[r, c])
        plt.tight_layout()
        plt.show()

    # 9) Cosine similarity across pooled embeddings
    if len(pooled) >= 2:
        names = list(pooled.keys())
        def cos(a, b):
            a = a.astype(np.float64); b = b.astype(np.float64)
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            return float(np.dot(a, b) / denom)

        print("\n--- Cosine Similarity (mean‑pooled audio token embeddings) ---")
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                s = cos(pooled[names[i]], pooled[names[j]])
                print(f"sim({names[i]} vs {names[j]}) = {s:.4f}")
    else:
        print("\n[INFO] Not enough valid embeddings to compute cosine similarity.")

    print("\n[DONE] Spatial encoder E2E test complete.")

if __name__ == "__main__":
    main()
