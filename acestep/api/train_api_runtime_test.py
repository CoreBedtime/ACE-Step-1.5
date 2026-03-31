"""Unit tests for temporary runtime component offload helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from acestep.api.train_api_runtime import RuntimeComponentManager


class _DummyModel:
    """Minimal module-like object exposing parameters and eval for tests."""

    def __init__(self) -> None:
        """Create a fixed parameter tensor and an eval spy."""

        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        self.eval = mock.Mock()

    def parameters(self):
        """Return a deterministic parameter iterator."""

        return iter([self._param])


class RuntimeComponentManagerTests(unittest.TestCase):
    """Behavior tests for temporary offload and restore management."""

    def test_offload_model_to_cpu_and_restore_llm_without_app_state(self) -> None:
        """The manager should offload the full model and restore it later."""

        model = _DummyModel()
        recursive_to_device = mock.Mock()
        release_memory = mock.Mock()
        handler = SimpleNamespace(
            model=model,
            dtype=torch.bfloat16,
            device="cuda:0",
            _recursive_to_device=recursive_to_device,
            _release_system_memory=release_memory,
        )
        llm = SimpleNamespace(
            llm_initialized=True,
            unload=mock.Mock(),
            initialize=mock.Mock(return_value=("restored", True)),
            last_init_params={"checkpoint_dir": "checkpoints", "lm_model_path": "model"},
        )

        manager = RuntimeComponentManager(handler=handler, llm=llm, app_state=None)

        with mock.patch.object(RuntimeComponentManager, "_device_of", return_value="cuda:0"), mock.patch.object(
            RuntimeComponentManager,
            "_dtype_of",
            return_value=torch.float16,
        ):
            manager.offload_model_to_cpu()

        self.assertTrue(manager.model_moved)
        recursive_to_device.assert_called_once_with(model, "cpu", torch.float16)
        release_memory.assert_called_once()

        manager.unload_llm()
        llm.unload.assert_called_once()

        manager.restore()

        self.assertFalse(manager.model_moved)
        self.assertFalse(manager.llm_unloaded)
        self.assertEqual(2, recursive_to_device.call_count)
        self.assertEqual(
            mock.call(model, "cuda:0", torch.float16),
            recursive_to_device.call_args_list[1],
        )
        model.eval.assert_called_once()
        llm.initialize.assert_called_once_with(
            checkpoint_dir="checkpoints",
            lm_model_path="model",
        )
        self.assertEqual(2, release_memory.call_count)


if __name__ == "__main__":
    unittest.main()
