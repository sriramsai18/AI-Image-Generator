import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestAppOptimizations(unittest.TestCase):

    def setUp(self):
        self.mock_pipe = MagicMock()
        self.mock_pipe.to.return_value = self.mock_pipe

        # Mock generated output image
        mock_image = Image.new("RGB", (512, 512), color="blue")
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        self.mock_pipe.return_value = mock_result

        # Preserve original sysconf if present
        self.orig_sysconf = getattr(os, "sysconf", None)

    def _mock_sysconf(self, ram_bytes):
        def sysconf_side_effect(key, *args, **kwargs):
            if key == "SC_PAGE_SIZE":
                return 4096
            if key == "SC_PHYS_PAGES":
                return ram_bytes // 4096
            if self.orig_sysconf:
                return self.orig_sysconf(key, *args, **kwargs)
            raise ValueError(f"Unknown sysconf key: {key}")

        return sysconf_side_effect

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_high_ram_cpu_disables_attention_slicing(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        mock_sysconf.side_effect = self._mock_sysconf(16 * 1024 * 1024 * 1024)  # 16 GB RAM
        mock_from_pretrained.return_value = self.mock_pipe

        import app
        importlib.reload(app)

        self.assertIsNotNone(app.pipe)
        self.mock_pipe.enable_attention_slicing.assert_not_called()
        self.mock_pipe.unet.to.assert_called_with(memory_format=app.torch.channels_last)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_low_ram_cpu_enables_attention_slicing(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        mock_sysconf.side_effect = self._mock_sysconf(2 * 1024 * 1024 * 1024)  # 2 GB RAM
        mock_from_pretrained.return_value = self.mock_pipe

        import app
        importlib.reload(app)

        self.assertIsNotNone(app.pipe)
        self.mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_runs_with_inference_mode(self, mock_cuda, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe

        inference_mode_active = []

        def side_effect(*args, **kwargs):
            import torch
            inference_mode_active.append(torch.is_inference_mode_enabled())
            mock_image = Image.new("RGB", (512, 512), color="blue")
            mock_result = MagicMock()
            mock_result.images = [mock_image]
            return mock_result

        self.mock_pipe.side_effect = side_effect

        import app
        importlib.reload(app)

        # Call generate_image
        image, info = app.generate_image(
            prompt="a glowing neon cat in space",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertIsNotNone(image)
        self.assertIn("✅ Generated in", info)
        self.mock_pipe.assert_called_once()
        self.assertTrue(inference_mode_active[0])

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_empty_prompt_returns_warning(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe

        import app
        importlib.reload(app)

        image, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)


if __name__ == "__main__":
    unittest.main()
