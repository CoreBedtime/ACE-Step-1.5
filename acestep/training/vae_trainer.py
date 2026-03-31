"""VAE decoder fine-tuning trainer for ACE-Step.

Trains only the decoder of ``diffusers.AutoencoderOobleck`` while keeping
the encoder frozen.  The loss is a weighted sum of:
  - Waveform L1 reconstruction loss
  - Multi-scale STFT magnitude loss (spectral convergence)

Uses Lightning Fabric for mixed-precision training and optional gradient
checkpointing.  Yields (step, loss, status) tuples so the Gradio UI can
stream live progress updates.
"""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

try:
    from lightning.fabric import Fabric
    from lightning.fabric.loggers import TensorBoardLogger

    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTNING_AVAILABLE = False
    logger.warning("Lightning Fabric not installed; VAE trainer will use basic loop.")

try:
    import bitsandbytes as bnb

    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

from acestep.training.configs import VaeTrainingConfig
from acestep.training.path_safety import safe_path
from acestep.training.vae_data_module import VaeDataModule

# ---------------------------------------------------------------------------
# Multi-scale STFT loss
# ---------------------------------------------------------------------------

_STFT_SCALES: List[Tuple[int, int, int]] = [
    # (n_fft, hop_length, win_length)
    (2048, 512, 2048),
    (1024, 256, 1024),
    (512, 128, 512),
    (256, 64, 256),
]


def _stft_magnitude(
    wav: torch.Tensor,
    n_fft: int,
    hop: int,
    win: int,
) -> torch.Tensor:
    """Compute STFT magnitude for a mono waveform.

    Args:
        wav: ``[B, T]`` float32 waveform.
        n_fft: FFT size.
        hop: Hop length.
        win: Window length.

    Returns:
        ``[B, n_fft//2 + 1, frames]`` magnitude spectrogram.
    """
    window = torch.hann_window(win, device=wav.device)
    stft = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        return_complex=True,
        normalized=False,
    )
    return stft.abs()


def multiscale_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scales: List[Tuple[int, int, int]] = _STFT_SCALES,
) -> torch.Tensor:
    """Compute averaged multi-scale STFT loss between two stereo waveforms.

    Each channel is processed independently; the result is averaged over
    channels and STFT scales.

    Args:
        pred: ``[B, 2, T]`` predicted waveform.
        target: ``[B, 2, T]`` target waveform.
        scales: List of (n_fft, hop, win) tuples.

    Returns:
        Scalar loss tensor.
    """
    total = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
    n_terms = 0
    signal_len = pred.shape[-1]
    for channel in range(pred.shape[1]):
        p_ch = pred[:, channel, :]
        t_ch = target[:, channel, :]
        for n_fft, hop, win in scales:
            # Skip scales whose half-window exceeds the signal length;
            # torch.stft would raise a padding error in that case.
            if win // 2 >= signal_len:
                continue
            # Ensure tensors are float32 for STFT
            p_mag = _stft_magnitude(p_ch.float(), n_fft, hop, win)
            t_mag = _stft_magnitude(t_ch.float(), n_fft, hop, win)

            # Spectral convergence component
            sc = torch.norm(t_mag - p_mag, p="fro") / (
                torch.norm(t_mag, p="fro") + 1e-8
            )
            # Log-magnitude component
            log_mag = F.l1_loss(
                torch.log(p_mag + 1e-7),
                torch.log(t_mag + 1e-7),
            )
            total = total + (sc + log_mag)
            n_terms += 1
    return total / max(n_terms, 1)


# ---------------------------------------------------------------------------
# Device helpers (mirrored from LoRA trainer for consistency)
# ---------------------------------------------------------------------------


def _normalize_device_type(device: Any) -> str:
    """Canonical device type string from a torch.device or string."""
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device)


def _select_compute_dtype(device_type: str) -> torch.dtype:
    """Pick the compute dtype for each accelerator."""
    if device_type in ("cuda", "xpu"):
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def _select_fabric_precision(device_type: str) -> str:
    """Pick Fabric precision plugin setting for each accelerator."""
    if device_type in ("cuda", "xpu"):
        return "bf16-mixed"
    if device_type == "mps":
        return "16-mixed"
    return "32-true"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_vae_decoder_weights(vae: nn.Module, output_dir: str) -> str:
    """Save the VAE decoder state-dict to *output_dir*/vae_decoder.safetensors.

    Args:
        vae: ``AutoencoderOobleck`` model (full VAE; only decoder weights saved).
        output_dir: Directory to write the checkpoint file into.

    Returns:
        Absolute path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "vae_decoder.safetensors")
    decoder = vae.decoder if hasattr(vae, "decoder") else vae
    # Unwrap Fabric wrapper if present
    while hasattr(decoder, "_forward_module"):
        decoder = decoder._forward_module

    state = {k: v.cpu() for k, v in decoder.state_dict().items()}
    try:
        from safetensors.torch import save_file

        save_file(state, out_path)
    except ImportError:
        # Fallback: save as .pt
        out_path = out_path.replace(".safetensors", ".pt")
        torch.save(state, out_path)
    logger.info(f"VAE decoder weights saved to {out_path}")
    return out_path


def load_vae_decoder_weights(vae: nn.Module, checkpoint_path: str) -> None:
    """Load VAE decoder weights from a safetensors or .pt checkpoint.

    Args:
        vae: ``AutoencoderOobleck`` model to patch in-place.
        checkpoint_path: Path to the checkpoint file.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    decoder = vae.decoder if hasattr(vae, "decoder") else vae
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing decoder keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        logger.warning(
            f"Unexpected decoder keys ({len(unexpected)}): {unexpected[:5]} ..."
        )
    logger.info(f"VAE decoder weights loaded from {checkpoint_path}")


def save_vae_training_checkpoint(
    vae: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    output_dir: str,
) -> None:
    """Save a full training checkpoint (weights + optimizer + scheduler state).

    Args:
        vae: The full VAE model.
        optimizer: Current optimizer.
        scheduler: Current LR scheduler.
        epoch: Current epoch (1-based).
        global_step: Current global optimizer step.
        output_dir: Directory to write files into.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_vae_decoder_weights(vae, output_dir)
    meta = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(meta, os.path.join(output_dir, "training_state.pt"))
    logger.info(f"VAE training checkpoint saved at epoch {epoch}, step {global_step}")


def load_vae_training_checkpoint(
    checkpoint_dir: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint, returning meta-information dict.

    Args:
        checkpoint_dir: Directory produced by :func:`save_vae_training_checkpoint`.
        optimizer: If provided, loads optimizer state in-place.
        scheduler: If provided, loads scheduler state in-place.
        device: Device to load tensors onto.

    Returns:
        Dict with keys ``epoch``, ``global_step``, ``loaded_optimizer``,
        ``loaded_scheduler``.
    """
    info: Dict[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "loaded_optimizer": False,
        "loaded_scheduler": False,
    }
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.isfile(state_path):
        logger.warning(f"No training_state.pt found in {checkpoint_dir}")
        return info

    map_loc = device if device is not None else "cpu"
    meta = torch.load(state_path, map_location=map_loc, weights_only=False)
    info["epoch"] = meta.get("epoch", 0)
    info["global_step"] = meta.get("global_step", 0)

    if optimizer is not None and "optimizer_state_dict" in meta:
        try:
            optimizer.load_state_dict(meta["optimizer_state_dict"])
            info["loaded_optimizer"] = True
        except Exception as exc:
            logger.warning(f"Could not restore optimizer state: {exc}")

    if scheduler is not None and "scheduler_state_dict" in meta:
        try:
            scheduler.load_state_dict(meta["scheduler_state_dict"])
            info["loaded_scheduler"] = True
        except Exception as exc:
            logger.warning(f"Could not restore scheduler state: {exc}")

    return info


# ---------------------------------------------------------------------------
# VAE Trainer
# ---------------------------------------------------------------------------


class VaeDecoderTrainer:
    """Fine-tune the ``AutoencoderOobleck`` decoder in ACE-Step.

    The encoder is kept frozen; only decoder parameters receive gradients.
    Training minimises a weighted sum of waveform L1 loss and multi-scale
    STFT loss, using Lightning Fabric for mixed precision.

    Args:
        vae: The ``AutoencoderOobleck`` model loaded from the handler.
        device: Torch device to train on.
        cfg: :class:`~acestep.training.configs.VaeTrainingConfig`.
    """

    def __init__(
        self,
        vae: nn.Module,
        device: torch.device,
        cfg: VaeTrainingConfig,
    ) -> None:
        self.vae = vae
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = _select_compute_dtype(self.device_type)
        self.cfg = cfg
        self.is_training = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        audio_dir: str,
        training_state: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Run the training loop, yielding (step, loss, status) tuples.

        Args:
            audio_dir: Directory of raw audio files for the dataset.
            training_state: Mutable dict for UI stop-signal communication.
                Checked for ``should_stop`` key each batch.
            resume_from: Optional checkpoint directory to resume from.

        Yields:
            ``(global_step, loss_value, status_message)`` tuples.
        """
        self.is_training = True
        try:
            yield from self._run(audio_dir, training_state, resume_from)
        except Exception as exc:
            logger.exception("VAE training failed")
            yield 0, 0.0, f"Training failed: {exc}"
        finally:
            self.is_training = False

    def stop(self) -> None:
        """Signal the training loop to stop after the current batch."""
        self.is_training = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters in-place."""
        encoder = getattr(self.vae, "encoder", None)
        if encoder is None:
            logger.warning("VAE has no .encoder attribute; nothing frozen.")
            return
        frozen = 0
        for p in encoder.parameters():
            p.requires_grad = False
            frozen += 1
        logger.info(f"Frozen {frozen} encoder parameter tensors.")

    def _compute_loss(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode → decode and compute reconstruction loss.

        Args:
            waveform: ``[B, 2, T]`` float32 waveform tensor on device.
            lengths: Optional valid lengths for each batch item.

        Returns:
            Scalar combined loss tensor (float32).
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        if lengths is not None:
            valid_lengths = [max(1, int(length)) for length in lengths.detach().cpu().tolist()]
            if waveform.shape[0] > 1:
                sample_losses = []
                for index, valid_length in enumerate(valid_lengths):
                    sample_waveform = waveform[index : index + 1, :, :valid_length]
                    sample_losses.append(self._compute_loss(sample_waveform))
                return torch.stack(sample_losses).mean()
            if valid_lengths:
                waveform = waveform[..., : valid_lengths[0]]

        if self.device_type in ("cuda", "xpu", "mps"):
            ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
        else:
            ctx = nullcontext()

        with ctx:
            # Encode — no grad through encoder (frozen)
            with torch.no_grad():
                latent_dist = self.vae.encode(waveform).latent_dist
                latents = latent_dist.sample()

            # Decode — decoder parameters receive gradients
            reconstructed = self.vae.decode(latents).sample

        # Trim to original length (decoder may output slightly different len)
        t_orig = waveform.shape[-1]
        t_out = reconstructed.shape[-1]
        if t_out > t_orig:
            reconstructed = reconstructed[..., :t_orig]
        elif t_out < t_orig:
            reconstructed = F.pad(reconstructed, (0, t_orig - t_out))

        # Ensure float32 for loss computation
        rec_f32 = reconstructed.float()
        wav_f32 = waveform.float()

        l1 = F.l1_loss(rec_f32, wav_f32)
        stft = multiscale_stft_loss(rec_f32, wav_f32)

        loss = self.cfg.l1_loss_weight * l1 + self.cfg.stft_loss_weight * stft
        return loss.float()

    def _build_optimizer(
        self,
        trainable_params: List[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
        """Build AdamW (8-bit if available on CUDA).

        Args:
            trainable_params: Parameters that require gradients.

        Returns:
            Configured optimizer instance.
        """
        kwargs = {
            "lr": self.cfg.learning_rate,
            "weight_decay": self.cfg.weight_decay,
        }
        if _HAS_BNB and self.device_type == "cuda":
            logger.info("Using bitsandbytes 8-bit AdamW for VAE training.")
            return bnb.optim.AdamW8bit(trainable_params, **kwargs)
        if self.device_type == "cuda":
            kwargs["fused"] = True
        return AdamW(trainable_params, **kwargs)

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build warmup + cosine annealing scheduler.

        Args:
            optimizer: Optimizer to schedule.
            total_steps: Total number of optimizer steps.

        Returns:
            :class:`~torch.optim.lr_scheduler.SequentialLR` instance.
        """
        warmup = min(self.cfg.warmup_steps, max(1, total_steps // 10))
        warmup_sched = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup),
            T_mult=1,
            eta_min=self.cfg.learning_rate * 0.01,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run(
        self,
        audio_dir: str,
        training_state: Optional[Dict],
        resume_from: Optional[str],
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Internal generator that drives the full training loop."""
        # Validate + safe-path audio dir
        try:
            audio_dir = safe_path(audio_dir)
        except ValueError as exc:
            yield 0, 0.0, f"Rejected unsafe audio directory: {exc}"
            return
        if not os.path.isdir(audio_dir):
            yield 0, 0.0, f"Audio directory not found: {audio_dir}"
            return

        # Reproducibility
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

        # Freeze encoder
        if self.cfg.freeze_encoder:
            self._freeze_encoder()

        # Dataset
        cfg = self.cfg
        try:
            dm = VaeDataModule(
                audio_dir=audio_dir,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                prefetch_factor=cfg.prefetch_factor,
                persistent_workers=cfg.persistent_workers,
                pin_memory_device=cfg.pin_memory_device,
                val_split=cfg.val_split,
            )
            dm.setup("fit")
        except ValueError as exc:
            yield 0, 0.0, f"Dataset error: {exc}"
            return

        n_train = len(dm.train_dataset)  # type: ignore[arg-type]
        yield 0, 0.0, f"Dataset ready: {n_train} audio files"

        if _LIGHTNING_AVAILABLE:
            yield from self._train_fabric(dm, training_state, resume_from)
        else:
            yield from self._train_basic(dm, training_state)

    def _train_fabric(
        self,
        dm: VaeDataModule,
        training_state: Optional[Dict],
        resume_from: Optional[str],
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Training loop using Lightning Fabric for mixed precision."""
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        precision = _select_fabric_precision(self.device_type)
        accelerator = (
            self.device_type
            if self.device_type in ("cuda", "xpu", "mps", "cpu")
            else "auto"
        )

        tb_logger = None
        try:
            tb_logger = TensorBoardLogger(root_dir=cfg.output_dir, name="logs")
        except (ModuleNotFoundError, Exception) as exc:
            logger.warning(f"TensorBoard logger unavailable: {exc}")

        fabric_kwargs: Dict[str, Any] = {
            "accelerator": accelerator,
            "devices": 1,
            "precision": precision,
        }
        if tb_logger is not None:
            fabric_kwargs["loggers"] = [tb_logger]
        fabric = Fabric(**fabric_kwargs)
        fabric.launch()

        yield 0, 0.0, f"Starting (device={self.device_type}, precision={precision})"

        # Keep decoder in fp32 for stable training; computation uses autocast
        vae_decoder = self.vae.decoder if hasattr(self.vae, "decoder") else self.vae
        vae_decoder = vae_decoder.to(dtype=torch.float32)

        # Collect only decoder parameters
        trainable_params = [p for p in vae_decoder.parameters() if p.requires_grad]
        if not trainable_params:
            yield 0, 0.0, "No trainable decoder parameters found."
            return

        n_trainable = sum(p.numel() for p in trainable_params)
        yield 0, 0.0, f"Training {n_trainable:,} decoder parameters"

        optimizer = self._build_optimizer(trainable_params)
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        steps_per_epoch = max(
            1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
        )
        total_steps = steps_per_epoch * cfg.max_epochs
        scheduler = self._build_scheduler(optimizer, total_steps)

        # Fabric setup
        vae_decoder, optimizer = fabric.setup(vae_decoder, optimizer)
        train_loader = fabric.setup_dataloaders(train_loader)

        # Resume
        start_epoch = 0
        global_step = 0
        if resume_from:
            try:
                resume_from = safe_path(resume_from)
                if os.path.isdir(resume_from):
                    yield 0, 0.0, f"Resuming from {resume_from} ..."
                    # Load decoder weights
                    sf_path = os.path.join(resume_from, "vae_decoder.safetensors")
                    pt_path = os.path.join(resume_from, "vae_decoder.pt")
                    ckpt_path = sf_path if os.path.isfile(sf_path) else pt_path
                    if os.path.isfile(ckpt_path):
                        # Unwrap Fabric wrapper before loading
                        raw_decoder = vae_decoder
                        while hasattr(raw_decoder, "_forward_module"):
                            raw_decoder = raw_decoder._forward_module
                        load_vae_decoder_weights(
                            type("_Shim", (), {"decoder": raw_decoder})(),
                            ckpt_path,
                        )
                    info = load_vae_training_checkpoint(
                        resume_from, optimizer, scheduler, device=self.device
                    )
                    start_epoch = info["epoch"]
                    global_step = info["global_step"]
                    status_parts = [f"Resumed epoch={start_epoch}, step={global_step}"]
                    if info["loaded_optimizer"]:
                        status_parts.append("optimizer OK")
                    if info["loaded_scheduler"]:
                        status_parts.append("scheduler OK")
                    yield 0, 0.0, ", ".join(status_parts)
                else:
                    yield (
                        0,
                        0.0,
                        f"Checkpoint dir not found: {resume_from}, starting fresh",
                    )
            except Exception as exc:
                logger.warning(f"Checkpoint load failed: {exc}")
                yield 0, 0.0, f"Warning: checkpoint load failed ({exc}), starting fresh"

        # EMA / tracking
        ema_loss: Optional[float] = None
        ema_alpha = 0.1
        best_val_loss = float("inf")

        if training_state is not None:
            for key in (
                "plot_steps",
                "plot_loss",
                "plot_ema",
                "plot_val_steps",
                "plot_val_loss",
            ):
                training_state.setdefault(key, [])
            training_state.setdefault("plot_best_step", None)

        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        vae_decoder.train()

        for epoch in range(start_epoch, cfg.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start = time.time()

            for batch in train_loader:
                # Stop signal
                if training_state and training_state.get("should_stop", False):
                    yield global_step, 0.0, "Training stopped by user"
                    return

                waveform = batch["waveform"].to(
                    self.device, non_blocking=self.device_type in ("cuda", "xpu")
                )
                lengths = batch.get("lengths")

                # The encoder is frozen but not set to eval — calling it in a
                # no_grad context inside _compute_loss handles this correctly.
                loss = self._compute_loss(waveform, lengths)
                loss = loss / cfg.gradient_accumulation_steps

                fabric.backward(loss)
                accumulated_loss += loss.item()
                accumulation_step += 1

                if accumulation_step >= cfg.gradient_accumulation_steps:
                    # Check for non-finite grads
                    bad_grads = sum(
                        1
                        for p in trainable_params
                        if p.grad is not None and not torch.isfinite(p.grad).all()
                    )
                    if bad_grads:
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_loss = 0.0
                        accumulation_step = 0
                        yield (
                            global_step,
                            float("nan"),
                            f"Non-finite grads ({bad_grads}); skipping step",
                        )
                        continue

                    fabric.clip_gradients(
                        vae_decoder,
                        optimizer,
                        max_norm=cfg.max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    avg_loss = accumulated_loss / accumulation_step
                    epoch_loss += avg_loss
                    num_updates += 1

                    if ema_loss is None:
                        ema_loss = avg_loss
                    else:
                        ema_loss = ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss

                    if global_step % cfg.log_every_n_steps == 0:
                        if training_state is not None:
                            training_state["plot_steps"].append(global_step)
                            training_state["plot_loss"].append(avg_loss)
                            training_state["plot_ema"].append(ema_loss)
                        fabric.log("train/loss", avg_loss, step=global_step)
                        fabric.log(
                            "train/lr",
                            scheduler.get_last_lr()[0],
                            step=global_step,
                        )
                        yield (
                            global_step,
                            avg_loss,
                            f"Epoch {epoch + 1}/{cfg.max_epochs}, "
                            f"Step {global_step}, Loss: {avg_loss:.5f}",
                        )

                    accumulated_loss = 0.0
                    accumulation_step = 0

            # Flush remainder
            if accumulation_step > 0:
                fabric.clip_gradients(
                    vae_decoder,
                    optimizer,
                    max_norm=cfg.max_grad_norm,
                    error_if_nonfinite=False,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accumulated_loss / accumulation_step
                epoch_loss += avg_loss
                num_updates += 1

                if ema_loss is None:
                    ema_loss = avg_loss
                else:
                    ema_loss = ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss

                if training_state is not None:
                    training_state["plot_steps"].append(global_step)
                    training_state["plot_loss"].append(avg_loss)
                    training_state["plot_ema"].append(ema_loss)

                accumulated_loss = 0.0
                accumulation_step = 0

            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            fabric.log("train/epoch_loss", avg_epoch_loss, step=epoch + 1)

            # Validation
            if dm.val_dataset is not None and dm.val_dataloader() is not None:
                vae_decoder.eval()
                total_val = 0.0
                n_val = 0
                with torch.no_grad():
                    for val_batch in dm.val_dataloader():  # type: ignore[union-attr]
                        v_wav = val_batch["waveform"].to(self.device)
                        v_loss = self._compute_loss(v_wav, val_batch.get("lengths"))
                        total_val += v_loss.item()
                        n_val += 1
                vae_decoder.train()
                val_loss = total_val / max(n_val, 1)
                if training_state is not None:
                    training_state["plot_val_steps"].append(global_step)
                    training_state["plot_val_loss"].append(val_loss)
                fabric.log("val/loss", val_loss, step=global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if training_state is not None:
                        training_state["plot_best_step"] = global_step
                    best_dir = os.path.join(cfg.output_dir, "checkpoints", "best")

                    # Unwrap for saving
                    raw_vae = _unwrap_vae(self.vae)
                    save_vae_training_checkpoint(
                        raw_vae, optimizer, scheduler, epoch + 1, global_step, best_dir
                    )
                    yield (
                        global_step,
                        avg_epoch_loss,
                        f"New best val loss {val_loss:.5f}, saved",
                    )

            # Periodic checkpoint
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                ckpt_dir = os.path.join(
                    cfg.output_dir,
                    "checkpoints",
                    f"epoch_{epoch + 1}_loss_{avg_epoch_loss:.5f}",
                )
                raw_vae = _unwrap_vae(self.vae)
                save_vae_training_checkpoint(
                    raw_vae, optimizer, scheduler, epoch + 1, global_step, ckpt_dir
                )
                yield (
                    global_step,
                    avg_epoch_loss,
                    f"Checkpoint saved at epoch {epoch + 1} ({epoch_time:.1f}s)",
                )
            else:
                yield (
                    global_step,
                    avg_epoch_loss,
                    f"Epoch {epoch + 1}/{cfg.max_epochs} done in {epoch_time:.1f}s, "
                    f"loss={avg_epoch_loss:.5f}",
                )

        # Final save
        final_dir = os.path.join(cfg.output_dir, "final")
        raw_vae = _unwrap_vae(self.vae)
        save_vae_training_checkpoint(
            raw_vae, optimizer, scheduler, cfg.max_epochs, global_step, final_dir
        )
        yield (
            global_step,
            avg_epoch_loss,
            f"Training complete. VAE decoder saved to {final_dir}",
        )

    def _train_basic(
        self,
        dm: VaeDataModule,
        training_state: Optional[Dict],
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Fallback training loop without Fabric."""
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        yield 0, 0.0, "Starting basic training loop (no Fabric)..."

        vae_decoder = self.vae.decoder if hasattr(self.vae, "decoder") else self.vae
        trainable_params = [p for p in vae_decoder.parameters() if p.requires_grad]
        if not trainable_params:
            yield 0, 0.0, "No trainable decoder parameters found."
            return

        optimizer = self._build_optimizer(trainable_params)
        train_loader = dm.train_dataloader()
        steps_per_epoch = max(
            1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
        )
        total_steps = steps_per_epoch * cfg.max_epochs
        scheduler = self._build_scheduler(optimizer, total_steps)

        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        vae_decoder.train()

        for epoch in range(cfg.max_epochs):
            epoch_loss = 0.0
            num_updates = 0

            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    yield global_step, 0.0, "Training stopped by user"
                    return

                waveform = batch["waveform"].to(self.device)
                loss = self._compute_loss(waveform, batch.get("lengths"))
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                accumulation_step += 1

                if accumulation_step >= cfg.gradient_accumulation_steps:
                    nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    avg_loss = accumulated_loss / accumulation_step
                    epoch_loss += avg_loss
                    num_updates += 1

                    if global_step % cfg.log_every_n_steps == 0:
                        yield (
                            global_step,
                            avg_loss,
                            (
                                f"Epoch {epoch + 1}/{cfg.max_epochs}, "
                                f"Step {global_step}, Loss: {avg_loss:.5f}"
                            ),
                        )
                    accumulated_loss = 0.0
                    accumulation_step = 0

            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            yield (
                global_step,
                avg_epoch_loss,
                (f"Epoch {epoch + 1}/{cfg.max_epochs} done, loss={avg_epoch_loss:.5f}"),
            )

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                ckpt_dir = os.path.join(
                    cfg.output_dir,
                    "checkpoints",
                    f"epoch_{epoch + 1}",
                )
                save_vae_decoder_weights(self.vae, ckpt_dir)
                yield global_step, avg_epoch_loss, f"Checkpoint saved: {ckpt_dir}"

        final_dir = os.path.join(cfg.output_dir, "final")
        save_vae_decoder_weights(self.vae, final_dir)
        yield global_step, avg_epoch_loss, f"Training complete. Saved to {final_dir}"


def _unwrap_vae(vae: nn.Module) -> nn.Module:
    """Return the innermost VAE module, unwrapping Fabric wrappers.

    Args:
        vae: VAE model, potentially wrapped by Lightning Fabric.

    Returns:
        Unwrapped :class:`~torch.nn.Module`.
    """
    unwrapped = vae
    while hasattr(unwrapped, "_forward_module"):
        unwrapped = unwrapped._forward_module
    return unwrapped
