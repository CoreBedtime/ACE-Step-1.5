"""Whole-file audio dataset helpers for VAE decoder fine-tuning.

Loads raw audio clips as whole-file stereo samples, resamples them to
48 kHz, and pads only at batch collation time. The dataset is intentionally
simple so the trainer can treat each source file as a single sample.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger
from torch.utils.data import Dataset

_TARGET_SR: int = 48_000
_TARGET_CHANNELS: int = 2
# The Oobleck encoder uses a 2x4x4x8x8 downsampling path, so clips shorter
# than 2048 samples cannot survive the first strided conv stack.
_MIN_AUDIO_SAMPLES: int = 2_048

_SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".aac",
    ".m4a",
)


def _safe_path(audio_dir: str) -> str:
    """Resolve the current module-level safe-path function.

    The public data-module wrapper re-exports ``safe_path`` so legacy tests and
    callers can patch the same attribute they used before the loader split.
    """
    from acestep.training.vae_data_module import safe_path as module_safe_path

    return module_safe_path(audio_dir)


def _scan_audio_files(root: str) -> List[str]:
    """Recursively scan *root* for supported audio files.

    Args:
        root: Validated absolute directory path.

    Returns:
        Sorted list of absolute file paths.
    """
    found: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in _SUPPORTED_EXTENSIONS:
                found.append(os.path.join(dirpath, fname))
    found.sort()
    return found


def _estimate_resampled_frames(path: str) -> Optional[int]:
    """Estimate the clip length after resampling to the target sample rate."""
    try:
        meta = torchaudio.info(path)
    except Exception as exc:
        logger.debug(f"Failed to read audio metadata for {path}: {exc}")
        return None

    if meta.sample_rate <= 0 or meta.num_frames <= 0:
        return None
    return int(math.ceil(meta.num_frames * _TARGET_SR / meta.sample_rate))


def _load_audio_file(path: str) -> Optional[torch.Tensor]:
    """Load and normalise a single audio file.

    Args:
        path: Absolute path to the audio file.

    Returns:
        Float32 waveform tensor ``[2, T]``, or ``None`` on error.
    """
    try:
        waveform, sr = torchaudio.load(path)

        if sr != _TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, _TARGET_SR)
            waveform = resampler(waveform)

        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        if waveform.shape[-1] == 0:
            return None

        return waveform.float()
    except Exception as exc:
        logger.debug(f"Failed to load audio file {path}: {exc}")
        return None


class VaeAudioDataset(Dataset):
    """Dataset of whole stereo audio files for VAE fine-tuning.

    Each item is a waveform tensor ``[2, T]`` plus its valid length.

    Args:
        audio_dir: Directory tree containing audio files.
    """

    def __init__(self, audio_dir: str) -> None:
        """Initialise dataset by scanning *audio_dir* and building file index."""
        validated = _safe_path(audio_dir)
        if not os.path.isdir(validated):
            raise ValueError(f"Not an existing directory: {audio_dir}")
        self.audio_dir = validated

        self._files = []
        skipped = 0
        for path in _scan_audio_files(validated):
            estimated_frames = _estimate_resampled_frames(path)
            if estimated_frames is None:
                skipped += 1
                logger.debug(f"Skipping unreadable audio file: {path}")
                continue
            if estimated_frames < _MIN_AUDIO_SAMPLES:
                skipped += 1
                logger.debug(
                    "Skipping too-short audio file ({} samples after resample): {}",
                    estimated_frames,
                    path,
                )
                continue
            self._files.append(path)

        if not self._files:
            raise ValueError(f"No audio files found under: {audio_dir}")

        if skipped:
            logger.warning(
                f"VaeAudioDataset skipped {skipped} unreadable or too-short files"
            )
        logger.info(f"VaeAudioDataset: {len(self._files)} files in {validated}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single whole-file audio sample.

        Returns:
            Dict with keys ``waveform`` and ``length``.
        """
        path = self._files[idx]
        waveform = _load_audio_file(path)
        if waveform is None:
            logger.warning(f"Using silence fallback for unreadable audio file: {path}")
            waveform = torch.zeros(_TARGET_CHANNELS, _MIN_AUDIO_SAMPLES)
        elif waveform.shape[-1] < _MIN_AUDIO_SAMPLES:
            pad = _MIN_AUDIO_SAMPLES - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad))
        return {
            "waveform": waveform,
            "length": torch.tensor(waveform.shape[-1], dtype=torch.long),
        }


def collate_vae_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a list of whole-file waveforms into a batch.

    Args:
        batch: List of per-sample dicts from :class:`VaeAudioDataset`.

    Returns:
        Dict with ``waveform`` → ``[B, 2, T]`` and ``lengths`` → ``[B]``.
    """
    lengths = torch.tensor(
        [int(sample["length"].item()) for sample in batch],
        dtype=torch.long,
    )
    max_len = int(lengths.max().item())
    waveforms: List[torch.Tensor] = []
    for sample in batch:
        waveform = sample["waveform"]
        pad = max_len - waveform.shape[-1]
        if pad > 0:
            waveform = F.pad(waveform, (0, pad))
        waveforms.append(waveform)
    stacked = torch.stack(waveforms, dim=0)
    return {"waveform": stacked, "lengths": lengths}
