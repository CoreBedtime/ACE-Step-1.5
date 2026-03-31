"""VAE decoder fine-tuning event handlers for the Gradio training UI.

Provides generator-based ``start_vae_training``, ``stop_vae_training``,
``scan_vae_dataset``, and ``export_vae_decoder`` functions consumed by
the Gradio wiring layer.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Dict, Iterator, Optional, Tuple

import torch
from loguru import logger

from acestep.training.path_safety import safe_path
from acestep.ui.gradio.events.training.training_utils import (
    _format_duration,
    _training_loss_figure,
)


def scan_vae_dataset(audio_dir: str) -> str:
    """Scan *audio_dir* and return a human-readable summary.

    Args:
        audio_dir: Path to the audio dataset directory.

    Returns:
        Status string for the dataset-info textbox.
    """
    if not audio_dir or not audio_dir.strip():
        return "Please enter an audio directory path."
    try:
        validated = safe_path(audio_dir.strip())
    except ValueError as exc:
        return f"Rejected unsafe path: {exc}"
    if not os.path.isdir(validated):
        return f"Directory not found: {validated}"

    _SUPPORTED = (".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a")
    count = 0
    for root, _, files in os.walk(validated):
        for f in files:
            if os.path.splitext(f)[1].lower() in _SUPPORTED:
                count += 1
    if count == 0:
        return f"No supported audio files found under: {validated}"
    return f"Found {count} audio file(s) in: {validated}"


def _release_idle_memory(device_type: str) -> None:
    """Release cached accelerator memory before starting VAE training."""
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif (
        device_type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()


def start_vae_training(
    audio_dir: str,
    dit_handler,
    val_split: float,
    learning_rate: float,
    train_epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    save_every_n_epochs: int,
    l1_loss_weight: float,
    stft_loss_weight: float,
    training_seed: int,
    output_dir: str,
    freeze_encoder: bool,
    resume_checkpoint_dir: str,
    training_state: Dict,
    progress=None,
) -> Iterator[Tuple[str, str, object, Dict]]:
    """Stream VAE training progress as ``(status, log, plot, state)`` tuples.

    Args:
        audio_dir: Directory of raw audio files.
        dit_handler: Initialised DiT handler (provides the VAE model).
        val_split: Validation fraction.
        learning_rate: AdamW learning rate.
        train_epochs: Maximum training epochs.
        batch_size: Training batch size.
        gradient_accumulation: Gradient accumulation steps.
        save_every_n_epochs: Checkpoint save frequency.
        l1_loss_weight: Weight for L1 waveform loss.
        stft_loss_weight: Weight for multi-scale STFT loss.
        training_seed: Random seed.
        output_dir: Output directory for checkpoints.
        freeze_encoder: Whether to freeze VAE encoder.
        resume_checkpoint_dir: Optional checkpoint directory to resume from.
        training_state: Shared mutable state dict for stop signalling.
        progress: Unused (reserved for Gradio progress arg).

    Yields:
        ``(status, log_text, loss_plot, training_state)`` tuples.
    """
    if not audio_dir or not audio_dir.strip():
        yield "Please enter an audio directory path.", "", None, training_state
        return

    try:
        audio_dir = safe_path(audio_dir.strip())
    except ValueError as exc:
        yield f"Rejected unsafe audio directory: {exc}", "", None, training_state
        return

    if not os.path.isdir(audio_dir):
        yield f"Audio directory not found: {audio_dir}", "", None, training_state
        return

    if dit_handler is None:
        yield (
            "Service not initialised. Please start the service first.",
            "",
            None,
            training_state,
        )
        return

    # Retrieve VAE from handler
    vae = _get_vae(dit_handler)
    if vae is None:
        yield (
            "VAE model not found on dit_handler. "
            "Ensure the service is fully initialised.",
            "",
            None,
            training_state,
        )
        return

    try:
        output_dir_safe = safe_path(
            output_dir.strip() if output_dir else "./vae_output"
        )
    except ValueError as exc:
        yield f"Rejected unsafe output directory: {exc}", "", None, training_state
        return

    training_state["is_training"] = True
    training_state["should_stop"] = False

    try:
        from acestep.training.configs import VaeTrainingConfig
        from acestep.training.vae_trainer import VaeDecoderTrainer

        # Detect device
        device_attr = getattr(dit_handler, "device", "cpu")
        if hasattr(device_attr, "type"):
            device_type = str(device_attr.type).lower()
        else:
            device_type = str(device_attr).split(":", 1)[0].lower()

        # Device-tuned dataloader defaults
        if device_type == "cuda":
            num_workers, pin_memory, prefetch_factor = 4, True, 2
            persistent_workers, pin_memory_device = True, "cuda"
        elif device_type == "xpu":
            num_workers, pin_memory, prefetch_factor = 4, True, 2
            persistent_workers, pin_memory_device = True, ""
        elif device_type == "mps":
            num_workers, pin_memory, prefetch_factor = 0, False, 2
            persistent_workers, pin_memory_device = False, ""
        else:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            num_workers = min(4, max(1, cpu_count // 2))
            pin_memory, prefetch_factor = False, 2
            persistent_workers = num_workers > 0
            pin_memory_device = ""

        _release_idle_memory(device_type)

        cfg = VaeTrainingConfig(
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            gradient_accumulation_steps=int(gradient_accumulation),
            max_epochs=int(train_epochs),
            save_every_n_epochs=int(save_every_n_epochs),
            seed=int(training_seed),
            output_dir=output_dir_safe,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            val_split=float(val_split),
            l1_loss_weight=float(l1_loss_weight),
            stft_loss_weight=float(stft_loss_weight),
            freeze_encoder=bool(freeze_encoder),
        )

        trainer = VaeDecoderTrainer(
            vae=vae,
            device=device_attr,
            cfg=cfg,
        )

        resume_from: Optional[str] = None
        if resume_checkpoint_dir and resume_checkpoint_dir.strip():
            try:
                candidate = safe_path(resume_checkpoint_dir.strip())
                if os.path.isdir(candidate):
                    resume_from = candidate
            except ValueError:
                logger.warning(f"Rejected unsafe resume path: {resume_checkpoint_dir}")

        log_lines: list[str] = []
        step_list: list[int] = []
        loss_list: list[float] = []
        start_time = time.time()
        training_failed = False
        failure_message = ""
        training_stopped = False
        stop_message = ""

        yield f"Starting VAE training from {audio_dir}...", "", None, training_state

        for step, loss, status in trainer.train(audio_dir, training_state, resume_from):
            status_text = str(status)
            status_lower = status_text.lower()
            if (
                status_text.startswith("❌")
                or "training failed" in status_lower
                or status_lower.startswith("error:")
                or "dataset error" in status_lower
            ):
                training_failed = True
                failure_message = status_text
            elif "stopped by user" in status_lower:
                training_stopped = True
                stop_message = status_text

            elapsed = time.time() - start_time
            time_str = f"Elapsed: {_format_duration(elapsed)}"
            display_status = f"{status_text}\n{time_str}"

            log_lines.append(status_text)
            if len(log_lines) > 15:
                log_lines = log_lines[-15:]

            if step > 0 and loss == loss:  # NaN filter
                step_list.append(step)
                loss_list.append(float(loss))

            plot = _training_loss_figure(training_state, step_list, loss_list)
            yield display_status, "\n".join(log_lines), plot, training_state

            if training_state.get("should_stop", False) or training_stopped:
                training_stopped = True
                break

        total_time = time.time() - start_time
        training_state["is_training"] = False
        final_plot = _training_loss_figure(training_state, step_list, loss_list)

        if training_failed:
            final_msg = (
                f"{failure_message}\nElapsed: {_format_duration(total_time)}"
            )
            yield final_msg, "\n".join(log_lines[-15:]), final_plot, training_state
            return

        if training_stopped:
            stop_msg = stop_message or "VAE training stopped by user."
            final_msg = f"{stop_msg}\nElapsed: {_format_duration(total_time)}"
            yield final_msg, "\n".join(log_lines[-15:]), final_plot, training_state
            return

        done_msg = f"VAE training finished. Total time: {_format_duration(total_time)}"
        yield done_msg, "\n".join(log_lines[-15:]), final_plot, training_state

    except Exception as exc:
        logger.exception("VAE training error")
        training_state["is_training"] = False
        yield f"Error: {exc}", str(exc), None, training_state


def stop_vae_training(training_state: Dict) -> Tuple[str, Dict]:
    """Signal the VAE training loop to stop.

    Args:
        training_state: Shared mutable state dict.

    Returns:
        ``(status_message, updated_state)`` tuple.
    """
    if not training_state.get("is_training", False):
        return "No VAE training in progress.", training_state
    training_state["should_stop"] = True
    return "Stopping VAE training...", training_state


def export_vae_decoder(export_path: str, vae_output_dir: str) -> str:
    """Copy the best/final VAE decoder checkpoint to *export_path*.

    Args:
        export_path: Destination directory.
        vae_output_dir: Training output directory to copy from.

    Returns:
        Status message string.
    """
    if not export_path or not export_path.strip():
        return "Please enter an export path."

    try:
        safe_export = safe_path(export_path.strip())
    except ValueError as exc:
        return f"Rejected unsafe export path: {exc}"

    try:
        safe_src_root = safe_path(
            vae_output_dir.strip() if vae_output_dir else "./vae_output"
        )
    except ValueError as exc:
        return f"Rejected unsafe output directory: {exc}"

    # Preference: final > best checkpoint > latest epoch checkpoint
    final_dir = os.path.join(safe_src_root, "final")
    best_dir = os.path.join(safe_src_root, "checkpoints", "best")
    ckpt_root = os.path.join(safe_src_root, "checkpoints")

    source: Optional[str] = None
    if os.path.isdir(final_dir):
        source = final_dir
    elif os.path.isdir(best_dir):
        source = best_dir
    elif os.path.isdir(ckpt_root):
        epoch_dirs = sorted(
            [d for d in os.listdir(ckpt_root) if d.startswith("epoch_")],
            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
        )
        if epoch_dirs:
            source = os.path.join(ckpt_root, epoch_dirs[-1])

    if source is None:
        return f"No VAE checkpoint found in: {safe_src_root}"

    try:
        parent = os.path.dirname(safe_export) or "."
        os.makedirs(parent, exist_ok=True)
        if os.path.exists(safe_export):
            shutil.rmtree(safe_export)
        shutil.copytree(source, safe_export)
        return f"VAE decoder exported to: {safe_export}"
    except Exception as exc:
        logger.exception("VAE export error")
        return f"Export failed: {exc}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_vae(dit_handler) -> object | None:
    """Extract the VAE model from *dit_handler*.

    Tries the dedicated handler VAE attributes first, then falls back to a
    nested ``model.vae`` pipeline layout when needed.

    Args:
        dit_handler: Initialised DiT handler instance.

    Returns:
        The VAE ``nn.Module``, or ``None`` if not found.
    """
    for attr in ("vae", "vae_model", "audio_vae"):
        candidate = getattr(dit_handler, attr, None)
        if candidate is not None:
            return candidate

    model = getattr(dit_handler, "model", None)
    if model is not None:
        vae = getattr(model, "vae", None)
        if vae is not None:
            return vae

    return None
