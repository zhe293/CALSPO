# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Tuple, Union
import io
import torch
import torchaudio

def process_audio(
    audio: Union[str, bytes, dict],
    processor=None,
    max_seconds: float = 15  # selected clips are at most 13.15 seconds
) -> Tuple[torch.Tensor, int]:
    """
    Load audio, convert to mono, resample, and clip to max_seconds.
    Supports file paths (str) and embedded bytes.
    """
    if isinstance(audio, dict):
        audio_path = audio.get("audio", audio)
    else:
        audio_path = audio

    try:
        # Load — handle both file paths and raw bytes
        if isinstance(audio_path, bytes):
            import numpy as np
            import scipy.io.wavfile as wav
            buf = io.BytesIO(audio_path)
            original_sr, audio_np = wav.read(buf)
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            elif audio_np.dtype == np.int32:
                audio_np = audio_np.astype(np.float32) / 2147483648.0
            elif audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            audio_data = torch.from_numpy(audio_np).unsqueeze(0)  # [1, samples]
        else:
            try:
                audio_data, original_sr = torchaudio.load(audio_path)
            except RuntimeError:
                # This P40 environment has torchaudio transforms but no audio I/O
                # backend. The selected HBA media are PCM WAV files.
                import numpy as np
                from scipy.io import wavfile

                original_sr, samples = wavfile.read(audio_path)
                if np.issubdtype(samples.dtype, np.integer):
                    samples = samples.astype(np.float32) / float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
                else:
                    samples = samples.astype(np.float32)
                audio_data = torch.from_numpy(samples.copy())
                if audio_data.ndim == 1:
                    audio_data = audio_data.unsqueeze(0)
                else:
                    audio_data = audio_data.T.contiguous()

        # Resample if needed
        if processor and hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
            target_sr = processor.feature_extractor.sampling_rate
        else:
            target_sr = 16000

        # HBA includes unusual sample rates (for example 24414 Hz). The
        # torchaudio rational resampler builds a huge filter for these rates.
        # Limit the original clip first, then use FFT resampling on CPU.
        if max_seconds:
            audio_data = audio_data[..., : int(max_seconds * original_sr)]
        if original_sr != target_sr:
            import numpy as np
            from scipy.signal import resample

            target_count = round(audio_data.shape[-1] * target_sr / original_sr)
            audio_data = torch.from_numpy(
                resample(audio_data.numpy(), target_count, axis=-1).astype(np.float32)
            )
        else:
            target_sr = original_sr

        # Convert to mono
        if audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0, keepdim=False)
        else:
            audio_data = audio_data.squeeze(0)

        # Clip to max_seconds
        if max_seconds:
            max_samples = int(max_seconds * target_sr)
            
            # print(f"Processing Audio {audio_path}, shape={audio_data.shape}, "
            #         f"sr={target_sr}, max_samples={max_samples}")
            # ValueError("Audio was processed")

            if audio_data.shape[0] > max_samples:
                audio_data = audio_data[:max_samples]

        return audio_data, target_sr

    except Exception as e:
        raise RuntimeError(f"Audio decoding failed for {audio_path}") from e


# def process_audio(
#     audio: Union[str, dict],
#     processor=None,
#     max_seconds: float = 10.0  # uniform duration in seconds
# ) -> Tuple[torch.Tensor, int]:
#     """
#     Load audio, convert to mono, resample, clip or pad to exactly max_seconds.
#     Returns:
#         audio_data (torch.Tensor): Tensor of shape [num_samples]
#         target_sr (int): Sampling rate used
#     """
#     # Determine target sampling rate
#     if processor and hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
#         target_sr = processor.feature_extractor.sampling_rate
#     else:
#         target_sr = 16000

#     # Expected number of samples
#     max_samples = int(max_seconds * target_sr)

#     # Get path if audio is a dict
#     if isinstance(audio, dict):
#         audio_path = audio.get("audio", audio)
#     else:
#         audio_path = audio

#     try:
#         # Load and resample if needed
#         audio_data, original_sr = torchaudio.load(audio_path)

#         if original_sr != target_sr:
#             resampler = torchaudio.transforms.Resample(original_sr, target_sr)
#             audio_data = resampler(audio_data)

#         # Convert to mono
#         if audio_data.shape[0] > 1:
#             audio_data = audio_data.mean(dim=0)
#         else:
#             audio_data = audio_data.squeeze(0)

#         # Clip or pad to uniform length
#         num_samples = audio_data.shape[0]
#         if num_samples > max_samples:
#             audio_data = audio_data[:max_samples]
#         elif num_samples < max_samples:
#             pad_length = max_samples - num_samples
#             audio_data = torch.nn.functional.pad(audio_data, (0, pad_length))

#         return audio_data, target_sr

#     except Exception as e:
#         print(f"[WARN] Error processing audio {audio_path}: {e}")
#         # Create uniform dummy audio (silence)
#         dummy_audio = torch.zeros((max_samples,), dtype=torch.float32)
#         return dummy_audio, target_sr
