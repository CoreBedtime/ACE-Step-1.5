"""Unit tests for VAE training helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from acestep.ui.gradio.events.training.vae_training import _get_vae


class VaeTrainingHelperTests(unittest.TestCase):
    """Behavior tests for VAE helper functions."""

    def test_get_vae_prefers_direct_handler_attribute(self) -> None:
        """The resolver should use the dedicated handler VAE before nested model VAE."""

        handler = SimpleNamespace(
            vae="direct-vae",
            model=SimpleNamespace(vae="nested-vae"),
        )

        self.assertEqual("direct-vae", _get_vae(handler))

    def test_get_vae_falls_back_to_nested_model_attribute(self) -> None:
        """The resolver should fall back to a nested model VAE when needed."""

        handler = SimpleNamespace(model=SimpleNamespace(vae="nested-vae"))

        self.assertEqual("nested-vae", _get_vae(handler))

    def test_get_vae_returns_none_when_missing(self) -> None:
        """The resolver should return None when no VAE is present."""

        handler = SimpleNamespace(model=SimpleNamespace(), vae=None)

        self.assertIsNone(_get_vae(handler))


if __name__ == "__main__":
    unittest.main()
