# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Qwen2.5Omni.
"""

import logging
import re
from typing import List, Optional, Union

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, VideoInput, make_batched_videos
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
import warnings

# --- START: Add these imports ---
# try:
    # feature_extraction_qwen2_audio_spatial should be in your python path
from .feature_extraction_qwen2_audio_spatial import SimplePowerVector
from ufb_banding.banding import BandingParams, BandingShape, LowerBandMode
from ufb_banding.ufb import TransformParams
_SPATIAL_FEAT_AVAILABLE = True
# except ImportError:
#     _SPATIAL_FEAT_AVAILABLE = False
#     warnings.warn(
#         "Could not import spatial feature extraction dependencies. "
#         "The Omni processor will not be able to handle multi-channel audio."
#     )
# --- END: Add these imports ---


class Qwen2_5_OmniVideosKwargs(VideosKwargs):
    fps: Optional[List[int]] = None
    use_audio_in_video: Optional[bool] = None
    seconds_per_chunk: Optional[float] = None
    position_id_per_seconds: Optional[int] = None
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2_5_OmniImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2_5OmniProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen2_5_OmniVideosKwargs
    images_kwargs: Qwen2_5_OmniImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "videos_kwargs": {
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 25,
            "use_audio_in_video": False,
            "min_pixels": 128 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "return_attention_mask": True,
        },
    }


class Qwen2_5OmniProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2.5Omni processor.
    [`Qwen2_5OmniProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`], [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2_5OmniProcessor.__call__`] and [`~Qwen2_5OmniProcessor.decode`] for more information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor.
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "Qwen2VLImageProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    valid_kwargs = ["chat_template"]

    def __init__(self, image_processor=None, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

        # --- START: Add this logic ---
        self.device = None
        self.spatial_feature_extractor = None
        if _SPATIAL_FEAT_AVAILABLE:
            self.spatial_feature_extractor = SimplePowerVector(
                params=BandingParams.Log(
                    dt_ms=96,
                    design_fs=self.feature_extractor.sampling_rate,
                    shape=BandingShape.SOFT,
                    transform_params=TransformParams.RaisedSine(),
                    lower_band_mode=LowerBandMode.VSV_HPF
                ),
                fs=self.feature_extractor.sampling_rate,
                nch=4, # First-Order Ambisonics
                stack_bands=False
            )
        # --- END: Add this logic ---

        # --- START: Add this method ---
        def to(self, device):
            if self.spatial_feature_extractor is not None:
                self.spatial_feature_extractor.to(device)
            self.device = device
            return self
    # --- END: Add this method ---

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. To prepare the vision inputs,
        this method forwards the `vision_infos` and `kwargs` arguments to Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`]
        if `vision_infos` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)

        if audio is not None:
            # output_kwargs["audio_kwargs"]["padding"] = "max_length"  # Support "max_length" padding only here
            # audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            # audio_inputs["feature_attention_mask"] = audio_inputs.pop(
            #     "attention_mask"
            # )  # rename feature_attention_mask to prevent conflicts later on

            
            # output_kwargs["audio_kwargs"]["padding"] = "max_length"
            # output_kwargs["audio_kwargs"]["return_attention_mask"] = True # Ensure mask is returned
        
            # audio_inputs = {}

            # # --- START: Add multichannel processing logic ---
            # # Pad short audio clips to prevent errors
            # MIN_SAMPLES = 1536
            # padded_audio = []
            # for i, arr in enumerate(audio):
            #     if arr.ndim != 2:
            #         raise ValueError(f"Audio at index {i} is not a 2D array (Channels, Samples).")
            #     if arr.shape[1] < MIN_SAMPLES:
            #         pad_width = MIN_SAMPLES - arr.shape[1]
            #         padding_spec = ((0, 0), (0, pad_width))
            #         padded_arr = np.pad(arr, pad_width=padding_spec, mode='constant', constant_values=0)
            #         padded_audio.append(padded_arr)
            #     else:
            #         padded_audio.append(arr)
            # audio = padded_audio

            # is_multichannel = isinstance(audio, list) and audio[0].ndim > 1 and audio[0].shape[0] > 1
            # print(f"\nDEBUG (Processor): Detected is_multichannel = {is_multichannel}")

            # print(f"DEBUG (Processor): _SPATIAL_FEAT_AVAILABLE = {_SPATIAL_FEAT_AVAILABLE}")
            # print(f"DEBUG (Processor): self.spatial_feature_extractor is None? {self.spatial_feature_extractor is None}")

            # --- Normalize to list of (C, T) float32 ---
            if not isinstance(audio, (list, tuple)):
                audio = [audio]

            norm_audio = []
            for i, a in enumerate(audio):
                a = np.asarray(a)
                if a.ndim == 1:
                    a = a[None, :]                     # (T,) -> (1, T)
                elif a.ndim == 2:
                    # common from soundfile: (T, C) with T >> C; flip to (C, T)
                    if a.shape[0] > a.shape[1]:
                        a = a.T
                else:
                    raise ValueError(f"Audio at index {i} must be 1D or 2D, got {a.ndim}D.")
                norm_audio.append(a.astype(np.float32, copy=False))
            audio = norm_audio  # list of (C, T)

            # --- Robust channel logic ---
            is_multichannel = any(a.shape[0] > 1 for a in audio)
            is_foa = is_multichannel and all(a.shape[0] == 4 for a in audio)

            print(f"\nDEBUG (Processor): Detected is_multichannel = {is_multichannel}")
            print(f"DEBUG (Processor): _SPATIAL_FEAT_AVAILABLE = {_SPATIAL_FEAT_AVAILABLE}")
            print(f"DEBUG (Processor): self.spatial_feature_extractor is None? {self.spatial_feature_extractor is None}")

            # Always prepare mono for Whisper: (list of 1-D np arrays)
            if is_foa:
                # FOA → pick W channel as mono (preserves scene energy best for Whisper)
                mono_list = [a[0] for a in audio]  # each is (T,)
            else:
                # Non-FOA → energy-preserving arithmetic mean
                mono_list = [a.mean(axis=0) for a in audio]  # each is (T,)

            # Ensure Whisper gets exactly list[ndarray(shape=(T,))]
            output_kwargs["audio_kwargs"]["padding"] = "max_length"
            output_kwargs["audio_kwargs"]["return_attention_mask"] = True
            mono_audio_inputs = self.feature_extractor(mono_list, **output_kwargs["audio_kwargs"])
            audio_inputs = dict(mono_audio_inputs)  # contains input_features and attention_mask

            # Rename for consistency with model code
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

            # Infer mel-frame lengths for special token replacement
            input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
            audio_lengths = iter((input_lengths - 2) // 2 + 1)


            # if is_multichannel and self.spatial_feature_extractor:
            #     if self.device is None:
            #         warnings.warn("OmniProcessor has not been moved to a device. Spatial feature extraction will be on CPU.")
                
            #     spatial_features_list = []
            #     for single_audio_np in audio:
            #         single_audio_tensor = torch.from_numpy(single_audio_np).float().permute(1, 0)
            #         if self.device is not None:
            #             single_audio_tensor = single_audio_tensor.to(self.device)
            #         with torch.no_grad():
            #             spatial_features_single = self.spatial_feature_extractor(single_audio_tensor.unsqueeze(0))
            #         spatial_features_list.append(spatial_features_single)

            #     spatial_features = torch.cat(spatial_features_list, dim=0)
            #     if spatial_features.dim() == 5 and spatial_features.shape[1] == 1:
            #         spatial_features = spatial_features.squeeze(1)
            #         print("DEBUG (Processor): spatial_features created.") # This print statement is inside the 'if' block
                
            #     # Model expects (B, C, F, T), extractor output is (B, T, C, F)
            #     sf = spatial_features
            #     if hasattr(sf, "names") and sf.names is not None:
            #         sf = sf.rename(None)
            #     audio_inputs["spatial_features"] = sf.permute(0, 2, 3, 1).contiguous().cpu()
            #     # audio_inputs["spatial_features"] = sf.permute(0, 2, 3, 1)[:, :4, :, :].contiguous().cpu()

            #     # Extract log-mel from W-channel (first channel) only for the main path
            #     w_channel_audio = [a[0] for a in audio]
            #     mono_audio_inputs = self.feature_extractor(w_channel_audio, **output_kwargs["audio_kwargs"])
            #     audio_inputs.update(mono_audio_inputs)
            
            # else:
            #     # Standard single-channel processing
            #     # if isinstance(audio[0], np.ndarray) and audio[0].ndim == 2:
            #     #     audio = [a[0] for a in audio] # Take first channel
            #     if is_multichannel:
            #          warnings.warn("Multi-channel audio detected, but spatial dependencies are not installed or extractor failed. Falling back to using the first channel only.")
            #          audio = [a[0] for a in audio]
            #     mono_audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            #     audio_inputs.update(mono_audio_inputs)
            # # --- END: Add multichannel processing logic ---

            # audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            # input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
            # audio_lengths = iter((input_lengths - 2) // 2 + 1)
            if is_foa and _SPATIAL_FEAT_AVAILABLE and self.spatial_feature_extractor is not None:
                # Expect model-side extractor: input (B, T, C), output (B, Tspat, Cspat, Fspat)
                tensor_list = []
                for a in audio:
                    # (C, T) -> (T, C), add batch
                    t = torch.from_numpy(a).float().permute(1, 0).unsqueeze(0)
                    if self.device is not None:
                        t = t.to(self.device)
                    tensor_list.append(t)
                spatial_in = torch.cat(tensor_list, dim=0)  # (B, T, C)

                with torch.no_grad():
                    spat = self.spatial_feature_extractor(spatial_in)  # expected (B, Tspat, Cspat, Fspat)

                # Convert to (B, Cspat, Fspat, Tspat) to match your audio encoder
                if spat.dim() == 4:
                    spat = spat.permute(0, 2, 3, 1).contiguous()
                audio_inputs["spatial_features"] = spat.detach().cpu()
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if images is not None:
            images_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        if videos is not None:
            videos = make_batched_videos(videos)
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            fps = [fps] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                self.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            video_grid_thw = iter(videos_inputs["video_grid_thw"])
            video_second_per_grid = iter(videos_inputs["video_second_per_grid"])
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
            image_grid_thw,
            video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
        )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        # Extend mm token length
        merge_length = self.image_processor.merge_size**2

        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length, 1)
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // merge_length
                        sample = sample.replace(self.video_token, "<|video_placeholder|>" * video_seq_length, 1)
                    else:
                        audio_token_indices = np.arange(next(audio_lengths))
                        curr_video_grid_thw = next(video_grid_thw)
                        height = curr_video_grid_thw[1] // self.image_processor.merge_size
                        width = curr_video_grid_thw[2] // self.image_processor.merge_size
                        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = np.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)
                        ).reshape(-1)
                        video_token_indices = (
                            video_token_indices * next(video_second_per_grid) * position_id_per_seconds
                        )

                        tokens_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = self.get_chunked_index(audio_token_indices, tokens_per_chunk)

                        placeholder_string = self.vision_bos_token + self.audio_bos_token
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                placeholder_string += "<|video_placeholder|>" * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                placeholder_string += "<|audio_placeholder|>" * audio_seq_length
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`np.ndarray`): A monotonically increasing list of token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).

        Returns:
            `List[Tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            if (
                conversation[0]["role"] != "system"
                or conversation[0]["content"][0]["text"]
                != "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            ):
                logging.warning(
                    "System prompt modified, audio output may not work as expected. "
                    + "Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'"
                )
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )


__all__ = ["Qwen2_5OmniProcessor"]
