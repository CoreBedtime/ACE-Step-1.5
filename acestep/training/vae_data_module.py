"""DataModule for VAE decoder fine-tuning.

This module keeps the public `VaeDataModule`, `VaeAudioDataset`, and
`collate_vae_batch` import paths stable while delegating the whole-file
audio loading logic to `vae_audio_dataset.py`.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from acestep.training.path_safety import safe_path
from acestep.training.vae_audio_dataset import VaeAudioDataset, collate_vae_batch

__all__ = ["VaeDataModule", "VaeAudioDataset", "collate_vae_batch", "safe_path"]


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
