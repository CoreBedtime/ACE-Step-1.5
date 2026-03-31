"""Dataset and DataModule for VAE decoder fine-tuning.

Loads raw audio files as whole-file stereo samples, resamples them to
48 kHz, and returns tensors suitable for decoder reconstruction training.
No chunking is applied; padding happens only at batch collation time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from acestep.training.path_safety import safe_path

# ACE-Step model parameters
_TARGET_SR: int = 48_000
_TARGET_CHANNELS: int = 2

_SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".aac",
    ".m4a",
)


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


def _load_audio_file(path: str) -> Optional[torch.Tensor]:
    """Load and normalise a single audio file.

    Args:
        path: Absolute path to the audio file.

    Returns:
        Float32 waveform tensor ``[2, T]``, or ``None`` on error.
    """
    try:
        waveform, sr = torchaudio.load(path)

        # Resample if needed
        if sr != _TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, _TARGET_SR)
            waveform = resampler(waveform)

        # Convert to stereo
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
        """Initialise dataset by scanning *audio_dir* and building file index.

        Args:
            audio_dir: Validated absolute path to audio directory.

        Raises:
            ValueError: If *audio_dir* does not exist or no audio found.
        """
        validated = safe_path(audio_dir)
        if not os.path.isdir(validated):
            raise ValueError(f"Not an existing directory: {audio_dir}")
        self.audio_dir = validated

        self._files = _scan_audio_files(validated)
        if not self._files:
            raise ValueError(f"No audio files found under: {audio_dir}")

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
            # Return silence rather than crashing the dataloader worker
            waveform = torch.zeros(_TARGET_CHANNELS, 1)
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


class VaeDataModule:
    """DataModule for :class:`VaeAudioDataset`.

    Manages train/val split and creates :class:`~torch.utils.data.DataLoader`
    instances for whole-file audio training.

    Args:
        audio_dir: Directory tree of audio files.
        batch_size: Training batch size.
        num_workers: DataLoader worker count.
        pin_memory: Enable pinned memory for faster GPU transfer.
        prefetch_factor: Prefetch depth per worker.
        persistent_workers: Keep workers alive between epochs.
        pin_memory_device: Device string for ``pin_memory_device`` kwarg.
        val_split: Fraction held out for validation.
    """

    def __init__(
        self,
        audio_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory_device: str = "",
        val_split: float = 0.0,
    ) -> None:
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device
        self.val_split = val_split

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build train/val splits.

        Args:
            stage: Ignored; provided for Lightning DataModule compatibility.
        """
        full = VaeAudioDataset(audio_dir=self.audio_dir)
        n_total = len(full)
        if self.val_split > 0.0 and n_total > 1:
            n_val = max(1, int(n_total * self.val_split))
            n_train = n_total - n_val
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full, [n_train, n_val]
            )
        else:
            self.train_dataset = full
            self.val_dataset = None

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Build a DataLoader with device-appropriate settings.

        Args:
            dataset: Dataset to wrap.
            shuffle: Whether to shuffle indices each epoch.

        Returns:
            Configured :class:`~torch.utils.data.DataLoader`.
        """
        effective_workers = self.num_workers
        prefetch = None if effective_workers == 0 else self.prefetch_factor
        persist = False if effective_workers == 0 else self.persistent_workers
        kwargs: dict = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=effective_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_vae_batch,
            drop_last=False,
            prefetch_factor=prefetch,
            persistent_workers=persist,
        )
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return the training :class:`~torch.utils.data.DataLoader`."""
        assert self.train_dataset is not None, "Call setup() first."
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return the validation DataLoader, or ``None`` if no val split."""
        if self.val_dataset is None:
            return None
        return self._make_loader(self.val_dataset, shuffle=False)

    def iter_files(self) -> Iterator[str]:
        """Yield status strings describing the dataset for UI display."""
        assert self.train_dataset is not None, "Call setup() first."
        n_train = len(self.train_dataset)
        n_val = len(self.val_dataset) if self.val_dataset is not None else 0
        yield f"Dataset ready: {n_train} train files, {n_val} val files"

    def iter_chunks(self) -> Iterator[str]:
        """Backward-compatible alias for :meth:`iter_files`."""
        yield from self.iter_files()
