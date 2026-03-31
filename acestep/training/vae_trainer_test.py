"""Unit tests for VAE decoder fine-tuning components.

Tests cover:
- VaeTrainingConfig validation
- VaeAudioDataset whole-file indexing and loading
- collate_vae_batch padding and length tracking
- multiscale_stft_loss shape and finite-value guarantees
- save/load_vae_decoder_weights round-trip
- VaeDecoderTrainer._compute_loss output properties
- VaeDecoderTrainer._build_optimizer selection logic
- scan_vae_dataset event handler
- export_vae_decoder event handler
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.nn as nn
import torchaudio


# ---------------------------------------------------------------------------
# Helper: bypass path-safety in tests
# ---------------------------------------------------------------------------
# Tests need to write to /tmp; patch safe_path to a permissive resolver.


def _identity_safe_path(user_path: str, *, base: str | None = None) -> str:
    """Permissive replacement for safe_path in unit tests."""
    if base is not None and not os.path.isabs(user_path):
        return os.path.normpath(os.path.join(base, user_path))
    return os.path.normpath(os.path.abspath(user_path))


# Context manager to patch safe_path in all training modules during tests
def _bypass_safe_path():
    """Return a patch context that makes safe_path permissive across modules."""
    targets = [
        "acestep.training.path_safety.safe_path",
        "acestep.training.vae_data_module.safe_path",
        "acestep.training.vae_trainer.safe_path",
        "acestep.ui.gradio.events.training.vae_training.safe_path",
    ]
    patches = [patch(t, side_effect=_identity_safe_path) for t in targets]

    class _MultiPatch:
        def __enter__(self):
            for p in patches:
                p.__enter__()
            return self

        def __exit__(self, *args):
            for p in reversed(patches):
                p.__exit__(*args)

    return _MultiPatch()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestVaeTrainingConfig(unittest.TestCase):
    """VaeTrainingConfig dataclass validation."""

    def test_defaults_are_valid(self) -> None:
        """Default config should construct without raising."""
        from acestep.training.configs import VaeTrainingConfig

        cfg = VaeTrainingConfig()
        self.assertEqual(cfg.freeze_encoder, True)
        self.assertEqual(cfg.batch_size, 1)

    def test_invalid_val_split_raises(self) -> None:
        """val_split >= 1.0 must raise ValueError."""
        from acestep.training.configs import VaeTrainingConfig

        with self.assertRaises(ValueError):
            VaeTrainingConfig(val_split=1.0)

    def test_to_dict_roundtrip(self) -> None:
        """to_dict() should include all expected keys."""
        from acestep.training.configs import VaeTrainingConfig

        cfg = VaeTrainingConfig(learning_rate=5e-5, max_epochs=10)
        d = cfg.to_dict()
        self.assertEqual(d["learning_rate"], 5e-5)
        self.assertEqual(d["max_epochs"], 10)
        self.assertIn("freeze_encoder", d)
        self.assertIn("stft_loss_weight", d)
        self.assertNotIn("chunk_duration_s", d)


# ---------------------------------------------------------------------------
# Multi-scale STFT loss tests
# ---------------------------------------------------------------------------


class TestMultiscaleStftLoss(unittest.TestCase):
    """multiscale_stft_loss output guarantees."""

    # Use waveforms long enough for the largest STFT scale (2048)
    _T = 4096

    def test_loss_is_finite_for_identical_inputs(self) -> None:
        """Loss should be a finite non-negative scalar when pred == target."""
        from acestep.training.vae_trainer import multiscale_stft_loss

        wav = torch.randn(2, 2, self._T)
        loss = multiscale_stft_loss(wav, wav)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))
        # Identical inputs → spectral convergence component is 0, log-mag
        # component is 0 → total loss should be ~0
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_loss_is_positive_for_different_inputs(self) -> None:
        """Loss between two random signals should be > 0."""
        from acestep.training.vae_trainer import multiscale_stft_loss

        torch.manual_seed(0)
        pred = torch.randn(2, 2, self._T)
        target = torch.randn(2, 2, self._T)
        loss = multiscale_stft_loss(pred, target)
        self.assertGreater(loss.item(), 0.0)

    def test_loss_works_with_batch_size_1(self) -> None:
        """Loss should work with B=1."""
        from acestep.training.vae_trainer import multiscale_stft_loss

        pred = torch.randn(1, 2, self._T)
        target = torch.randn(1, 2, self._T)
        loss = multiscale_stft_loss(pred, target)
        self.assertTrue(torch.isfinite(loss))

    def test_loss_skips_oversized_scales(self) -> None:
        """Loss should not crash for very short waveforms (small-scale fallback)."""
        from acestep.training.vae_trainer import (
            multiscale_stft_loss,
            _STFT_SCALES,
        )

        # Use a scale list with only scales that fit
        tiny_T = 512
        small_scales = [(n, h, w) for n, h, w in _STFT_SCALES if n <= tiny_T]
        if not small_scales:
            self.skipTest("No scales fit the tiny waveform; skip.")
        pred = torch.randn(1, 2, tiny_T)
        target = torch.randn(1, 2, tiny_T)
        loss = multiscale_stft_loss(pred, target, scales=small_scales)
        self.assertTrue(torch.isfinite(loss))


# ---------------------------------------------------------------------------
# VAE dataset tests
# ---------------------------------------------------------------------------


def _write_dummy_wav(path: str, sr: int = 48000, duration_s: float = 2.0) -> None:
    """Write a minimal silent WAV to *path*."""
    n_frames = int(sr * duration_s)
    waveform = torch.zeros(2, n_frames)
    torchaudio.save(path, waveform, sr, format="wav")


class TestVaeAudioDataset(unittest.TestCase):
    """VaeAudioDataset whole-file indexing and item loading."""

    def test_single_file_yields_one_item(self) -> None:
        """A single file should yield exactly one dataset item."""
        from acestep.training.vae_data_module import VaeAudioDataset

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                _write_dummy_wav(os.path.join(tmp, "test.wav"), duration_s=2.0)
                ds = VaeAudioDataset(audio_dir=tmp)
                self.assertEqual(len(ds), 1)

    def test_two_files_yield_two_items(self) -> None:
        """Two audio files should yield two dataset items."""
        from acestep.training.vae_data_module import VaeAudioDataset

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                _write_dummy_wav(os.path.join(tmp, "a.wav"), duration_s=2.0)
                _write_dummy_wav(os.path.join(tmp, "b.wav"), duration_s=3.0)
                ds = VaeAudioDataset(audio_dir=tmp)
                self.assertEqual(len(ds), 2)

    def test_too_short_file_is_skipped(self) -> None:
        """Files shorter than the VAE minimum should not enter the dataset."""
        from acestep.training.vae_data_module import VaeAudioDataset

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                _write_dummy_wav(os.path.join(tmp, "tiny.wav"), duration_s=0.001)
                _write_dummy_wav(os.path.join(tmp, "ok.wav"), duration_s=2.0)
                ds = VaeAudioDataset(audio_dir=tmp)
                self.assertEqual(len(ds), 1)

    def test_item_has_correct_shape(self) -> None:
        """Each item should contain the full loaded waveform."""
        from acestep.training.vae_data_module import VaeAudioDataset

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                _write_dummy_wav(os.path.join(tmp, "t.wav"), duration_s=5.0)
                ds = VaeAudioDataset(audio_dir=tmp)
                item = ds[0]
                expected_samples = int(5.0 * 48000)
                self.assertEqual(item["waveform"].shape, (2, expected_samples))

    def test_empty_directory_raises(self) -> None:
        """Empty directory should raise ValueError."""
        from acestep.training.vae_data_module import VaeAudioDataset

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    VaeAudioDataset(audio_dir=tmp)

    def test_nonexistent_directory_raises(self) -> None:
        """Non-existent directory should raise ValueError."""
        from acestep.training.vae_data_module import VaeAudioDataset

        with _bypass_safe_path():
            with self.assertRaises(ValueError):
                VaeAudioDataset(audio_dir="/nonexistent/path/xyz123")


class TestCollateVaeBatch(unittest.TestCase):
    """collate_vae_batch output shape and length tracking."""

    def test_batch_padded_and_lengths_preserved(self) -> None:
        """Variable-length items should be padded and lengths preserved."""
        from acestep.training.vae_data_module import collate_vae_batch

        short = torch.zeros(2, 24000)
        long = torch.zeros(2, 48000)
        batch = [
            {"waveform": short, "length": torch.tensor(24000)},
            {"waveform": long, "length": torch.tensor(48000)},
        ]
        out = collate_vae_batch(batch)
        self.assertEqual(out["waveform"].shape, (2, 2, 48000))
        self.assertTrue(torch.equal(out["lengths"], torch.tensor([24000, 48000])))


# ---------------------------------------------------------------------------
# Checkpoint save/load tests
# ---------------------------------------------------------------------------


class _DummyDecoder(nn.Module):
    """Minimal decoder stub for checkpoint round-trip tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _DummyVAE(nn.Module):
    """Minimal VAE stub (encoder + decoder)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = _DummyDecoder()


class TestVaeCheckpoints(unittest.TestCase):
    """save/load_vae_decoder_weights round-trip."""

    def test_save_and_load_safetensors(self) -> None:
        """State dict should survive a save→load round-trip."""
        from acestep.training.vae_trainer import (
            load_vae_decoder_weights,
            save_vae_decoder_weights,
        )

        vae = _DummyVAE()
        with tempfile.TemporaryDirectory() as tmp:
            saved_path = save_vae_decoder_weights(vae, tmp)
            self.assertTrue(os.path.isfile(saved_path))

            # Mutate weights, then restore
            vae2 = _DummyVAE()
            nn.init.constant_(vae2.decoder.linear.weight, 99.0)
            load_vae_decoder_weights(vae2, saved_path)

            self.assertTrue(
                torch.allclose(
                    vae.decoder.linear.weight,
                    vae2.decoder.linear.weight,
                )
            )

    def test_load_missing_file_raises(self) -> None:
        """Loading from a non-existent path should raise FileNotFoundError."""
        from acestep.training.vae_trainer import load_vae_decoder_weights

        vae = _DummyVAE()
        with self.assertRaises(FileNotFoundError):
            load_vae_decoder_weights(vae, "/nonexistent/vae_decoder.safetensors")


# ---------------------------------------------------------------------------
# VaeDecoderTrainer._compute_loss tests
# ---------------------------------------------------------------------------

# Use a waveform length that fits all STFT scales (n_fft <= 4096)
_COMPUTE_LOSS_T = 4096


class _MinimalVAE(nn.Module):
    """Tiny VAE-alike that exposes .encode()/.decode() returning named tuples."""

    def __init__(self, output_samples: int = _COMPUTE_LOSS_T) -> None:
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self._output_samples = output_samples

    class _LatentDist:
        def __init__(self, z: torch.Tensor) -> None:
            self._z = z

        def sample(self) -> torch.Tensor:
            return self._z

    class _EncOut:
        def __init__(self, z: torch.Tensor) -> None:
            self.latent_dist = _MinimalVAE._LatentDist(z)

    class _DecOut:
        def __init__(self, x: torch.Tensor) -> None:
            self.sample = x

    def encode(self, wav: torch.Tensor) -> "_MinimalVAE._EncOut":
        # Produce a latent with downsampled time axis
        z = torch.zeros(wav.shape[0], 4, max(1, wav.shape[-1] // 64))
        return self._EncOut(z)

    def decode(self, z: torch.Tensor) -> "_MinimalVAE._DecOut":
        B = z.shape[0]
        out = torch.zeros(B, 2, self._output_samples)
        return self._DecOut(out)


class TestVaeDecoderTrainerComputeLoss(unittest.TestCase):
    """VaeDecoderTrainer._compute_loss basic contract."""

    def _make_trainer(self) -> Any:
        from acestep.training.configs import VaeTrainingConfig
        from acestep.training.vae_trainer import VaeDecoderTrainer

        vae = _MinimalVAE(output_samples=_COMPUTE_LOSS_T)
        cfg = VaeTrainingConfig(max_epochs=1)
        return VaeDecoderTrainer(vae=vae, device=torch.device("cpu"), cfg=cfg)

    def test_loss_is_finite_scalar(self) -> None:
        """_compute_loss must return a finite scalar tensor."""
        trainer = self._make_trainer()
        waveform = torch.randn(1, 2, _COMPUTE_LOSS_T)
        loss = trainer._compute_loss(waveform)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))

    def test_loss_is_non_negative(self) -> None:
        """L1 + STFT loss should always be >= 0."""
        trainer = self._make_trainer()
        waveform = torch.randn(1, 2, _COMPUTE_LOSS_T)
        loss = trainer._compute_loss(waveform)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_loss_uses_lengths_for_padded_batches(self) -> None:
        """Padded batches should still yield a finite masked loss."""
        trainer = self._make_trainer()
        waveform = torch.randn(2, 2, _COMPUTE_LOSS_T)
        lengths = torch.tensor([1024, _COMPUTE_LOSS_T], dtype=torch.long)
        loss = trainer._compute_loss(waveform, lengths)
        self.assertTrue(torch.isfinite(loss))


# ---------------------------------------------------------------------------
# scan_vae_dataset handler tests
# ---------------------------------------------------------------------------


class TestScanVaeDataset(unittest.TestCase):
    """scan_vae_dataset edge cases."""

    def test_empty_string_returns_prompt(self) -> None:
        from acestep.ui.gradio.events.training.vae_training import scan_vae_dataset

        result = scan_vae_dataset("")
        self.assertIn("enter", result.lower())

    def test_nonexistent_dir_returns_not_found(self) -> None:
        from acestep.ui.gradio.events.training.vae_training import scan_vae_dataset

        with _bypass_safe_path():
            result = scan_vae_dataset("/definitely/not/real/xyz999")
        self.assertIn("not found", result.lower())

    def test_directory_with_wav_returns_count(self) -> None:
        from acestep.ui.gradio.events.training.vae_training import scan_vae_dataset

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                _write_dummy_wav(os.path.join(tmp, "a.wav"))
                _write_dummy_wav(os.path.join(tmp, "b.wav"))
                result = scan_vae_dataset(tmp)
        self.assertIn("2", result)

    def test_empty_directory_reports_no_files(self) -> None:
        from acestep.ui.gradio.events.training.vae_training import scan_vae_dataset

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                result = scan_vae_dataset(tmp)
        self.assertIn("No supported", result)


# ---------------------------------------------------------------------------
# export_vae_decoder handler tests
# ---------------------------------------------------------------------------


class TestExportVaeDecoder(unittest.TestCase):
    """export_vae_decoder path-safety and happy-path tests."""

    def test_empty_export_path_returns_prompt(self) -> None:
        from acestep.ui.gradio.events.training.vae_training import export_vae_decoder

        result = export_vae_decoder("", "./vae_output")
        self.assertIn("enter", result.lower())

    def test_missing_source_returns_not_found(self) -> None:
        from acestep.ui.gradio.events.training.vae_training import export_vae_decoder

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                result = export_vae_decoder(
                    os.path.join(tmp, "export"),
                    os.path.join(tmp, "vae_out"),  # no checkpoints here
                )
        self.assertIn("No VAE checkpoint", result)

    def test_copies_final_dir_to_export_path(self) -> None:
        from acestep.training.vae_trainer import save_vae_decoder_weights
        import acestep.ui.gradio.events.training.vae_training as _vae_mod

        vae = _DummyVAE()
        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                output_dir = os.path.join(tmp, "vae_out")
                final_dir = os.path.join(output_dir, "final")
                save_vae_decoder_weights(vae, final_dir)

                export_target = os.path.join(tmp, "exported")
                result = _vae_mod.export_vae_decoder(export_target, output_dir)
                # Assertions must be inside the tempdir context
                self.assertIn("exported", result.lower())
                self.assertTrue(os.path.isdir(export_target))


# ---------------------------------------------------------------------------
# start_vae_training wrapper tests
# ---------------------------------------------------------------------------


class TestStartVaeTrainingWrapper(unittest.TestCase):
    """start_vae_training status-stream handling."""

    def _run_training(self, trainer_cls: type) -> tuple[list[tuple[object, ...]], dict]:
        """Run the VAE training wrapper with a fake trainer."""

        from acestep.ui.gradio.events.training.vae_training import start_vae_training

        with _bypass_safe_path():
            with tempfile.TemporaryDirectory() as tmp:
                audio_dir = os.path.join(tmp, "audio")
                os.makedirs(audio_dir, exist_ok=True)
                output_dir = os.path.join(tmp, "vae_out")
                state = {"is_training": False, "should_stop": False}
                handler = type("Handler", (), {"device": "cpu"})()

                with patch(
                    "acestep.training.vae_trainer.VaeDecoderTrainer",
                    trainer_cls,
                ), patch(
                    "acestep.ui.gradio.events.training.vae_training._get_vae",
                    return_value=object(),
                ):
                    outputs = list(
                        start_vae_training(
                            audio_dir=audio_dir,
                            dit_handler=handler,
                            val_split=0.0,
                            learning_rate=1e-4,
                            train_epochs=1,
                            batch_size=1,
                            gradient_accumulation=1,
                            save_every_n_epochs=1,
                            l1_loss_weight=1.0,
                            stft_loss_weight=1.0,
                            training_seed=42,
                            output_dir=output_dir,
                            freeze_encoder=True,
                            resume_checkpoint_dir="",
                            training_state=state,
                        )
                    )

        return outputs, state

    def test_success_path_finishes_normally(self) -> None:
        """A normal trainer stream should end with a finished message."""

        class _SuccessTrainer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def train(self, audio_dir, training_state, resume_from):
                del audio_dir, training_state, resume_from
                yield 1, 0.25, "Epoch 1/1, Step 1, Loss: 0.25000"

        outputs, state = self._run_training(_SuccessTrainer)
        self.assertGreaterEqual(len(outputs), 3)
        self.assertIn("finished", str(outputs[-1][0]).lower())
        self.assertFalse(state["is_training"])

    def test_failure_path_does_not_report_finished(self) -> None:
        """A failed trainer stream should not emit a finished message."""

        class _FailedTrainer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def train(self, audio_dir, training_state, resume_from):
                del audio_dir, training_state, resume_from
                yield 0, 0.0, "Training failed: boom"

        outputs, state = self._run_training(_FailedTrainer)
        self.assertGreaterEqual(len(outputs), 3)
        self.assertIn("failed", str(outputs[-1][0]).lower())
        self.assertNotIn("finished", str(outputs[-1][0]).lower())
        self.assertFalse(state["is_training"])

    def test_stop_path_does_not_report_finished(self) -> None:
        """A stopped trainer stream should end with a stop message."""

        class _StoppedTrainer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def train(self, audio_dir, training_state, resume_from):
                del audio_dir, resume_from
                training_state["should_stop"] = True
                yield 1, 0.25, "Training stopped by user"

        outputs, state = self._run_training(_StoppedTrainer)
        self.assertGreaterEqual(len(outputs), 3)
        self.assertIn("stopped", str(outputs[-1][0]).lower())
        self.assertNotIn("finished", str(outputs[-1][0]).lower())
        self.assertFalse(state["is_training"])


if __name__ == "__main__":
    unittest.main()
