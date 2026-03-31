"""Unit tests for VAE training wiring and temporary runtime offload."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from acestep.ui.gradio.events.wiring.training_vae_wiring import _build_vae_training_wrapper


class _FakeManager:
    """Record offload and restore operations for wiring tests."""

    def __init__(self, events: list[object]) -> None:
        """Store the shared event list used by a test case."""

        self.events = events

    def offload_model_to_cpu(self) -> None:
        """Record that the main model was offloaded."""

        self.events.append("offload_model")

    def offload_text_encoder_to_cpu(self) -> None:
        """Record that the text encoder was offloaded."""

        self.events.append("offload_text_encoder")

    def unload_llm(self) -> None:
        """Record that the LLM was unloaded."""

        self.events.append("unload_llm")

    def restore(self) -> None:
        """Record that runtime components were restored."""

        self.events.append("restore")


class VaeTrainingWiringTests(unittest.TestCase):
    """Behavior tests for the VAE training wrapper."""

    def test_wrapper_offloads_before_starting_training(self) -> None:
        """The wrapper should offload runtime components before yielding training data."""

        events: list[object] = []
        training_state = {"is_training": False, "should_stop": False}

        def normalize(state: object) -> dict[str, object]:
            """Return the provided state mapping unchanged."""

            return state if isinstance(state, dict) else training_state

        def fake_start_vae_training(**kwargs):
            """Yield one progress update while recording call order."""

            events.append(("train_start", kwargs["audio_dir"]))
            yield ("status", "log", "plot", kwargs["training_state"])
            events.append("train_done")

        with mock.patch(
            "acestep.ui.gradio.events.wiring.training_vae_wiring.RuntimeComponentManager",
            side_effect=lambda *args, **kwargs: _FakeManager(events),
        ), mock.patch(
            "acestep.ui.gradio.events.wiring.training_vae_wiring.start_vae_training",
            side_effect=fake_start_vae_training,
        ):
            wrapper = _build_vae_training_wrapper(
                SimpleNamespace(),
                SimpleNamespace(),
                normalize,
            )
            results = list(
                wrapper(
                    "/tmp/audio",
                    0.1,
                    1e-4,
                    1,
                    1,
                    1,
                    1,
                    1.0,
                    1.0,
                    42,
                    "/tmp/out",
                    True,
                    "",
                    training_state,
                )
            )

        self.assertEqual(
            [
                "offload_model",
                "offload_text_encoder",
                "unload_llm",
                ("train_start", "/tmp/audio"),
                "train_done",
                "restore",
            ],
            events,
        )
        self.assertEqual([("status", "log", "plot", training_state)], results)

    def test_wrapper_restores_runtime_after_training_error(self) -> None:
        """The wrapper should restore runtime components even when training raises."""

        events: list[object] = []
        training_state = {"is_training": False, "should_stop": False}

        def normalize(state: object) -> dict[str, object]:
            """Return the provided state mapping unchanged."""

            return state if isinstance(state, dict) else training_state

        def failing_start_vae_training(**kwargs):
            """Yield once and then raise to simulate a training failure."""

            events.append("train_start")
            yield ("status", "log", "plot", kwargs["training_state"])
            raise RuntimeError("boom")

        with mock.patch(
            "acestep.ui.gradio.events.wiring.training_vae_wiring.RuntimeComponentManager",
            side_effect=lambda *args, **kwargs: _FakeManager(events),
        ), mock.patch(
            "acestep.ui.gradio.events.wiring.training_vae_wiring.start_vae_training",
            side_effect=failing_start_vae_training,
        ):
            wrapper = _build_vae_training_wrapper(
                SimpleNamespace(),
                SimpleNamespace(),
                normalize,
            )
            generator = wrapper(
                "/tmp/audio",
                0.1,
                1e-4,
                1,
                1,
                1,
                1,
                1.0,
                1.0,
                42,
                "/tmp/out",
                True,
                "",
                training_state,
            )

            self.assertEqual(("status", "log", "plot", training_state), next(generator))
            self.assertEqual(("❌ Error: boom", "boom", None, training_state), next(generator))
            with self.assertRaises(StopIteration):
                next(generator)

        self.assertEqual(
            [
                "offload_model",
                "offload_text_encoder",
                "unload_llm",
                "train_start",
                "restore",
            ],
            events,
        )


if __name__ == "__main__":
    unittest.main()
