import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Ensure gradio doesn't fail networking checks in restricted environments
os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
_real_sysconf = getattr(os, "sysconf", None)


class TestAppPerformanceOptimizations(unittest.TestCase):

    def setUp(self):
        # Default mock mode off unless specified
        os.environ["MOCK_MODE"] = "0"

    def _reload_app(self):
        if "app" in sys.modules:
            return importlib.reload(sys.modules["app"])
        else:
            return importlib.import_module("app")

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_pipeline_optimization_initialization(self, mock_from_pretrained):
        # Setup fresh mock pipeline with mock unet and vae
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae
        mock_from_pretrained.return_value = mock_pipe

        app = self._reload_app()

        # Verify channels_last format was applied to UNet and VAE
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)
        self.assertIsNotNone(app)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("os.sysconf", create=True)
    def test_attention_slicing_on_low_memory(self, mock_sysconf, mock_from_pretrained, mock_cuda):
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            elif name == "SC_PHYS_PAGES":
                # Returns ~2GB RAM (<4GB)
                return 2 * 1024 * 1024 * 1024 // 4096
            if _real_sysconf:
                return _real_sysconf(name)
            raise ValueError(f"Unknown sysconf {name}")

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        self._reload_app()

        mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("os.sysconf", create=True)
    def test_attention_slicing_skipped_on_high_memory(self, mock_sysconf, mock_from_pretrained, mock_cuda):
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            elif name == "SC_PHYS_PAGES":
                # Returns 16GB RAM (>=4GB)
                return 16 * 1024 * 1024 * 1024 // 4096
            if _real_sysconf:
                return _real_sysconf(name)
            raise ValueError(f"Unknown sysconf {name}")

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        self._reload_app()

        mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        app = self._reload_app()
        img, status = app.generate_image("   ", "neg", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_inference_mode(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_result = MagicMock()
        mock_result.images = ["fake_image_object"]

        def check_inference_mode(**kwargs):
            # Verify torch.is_inference_mode_enabled() is True when pipe is called
            self.assertTrue(torch.is_inference_mode_enabled())
            return mock_result

        mock_pipe.side_effect = check_inference_mode
        mock_from_pretrained.return_value = mock_pipe

        app = self._reload_app()

        img, status = app.generate_image("a neon cyberpunk runner", "blurry", 20, 7.0, 512, 512, 1234)
        self.assertEqual(img, "fake_image_object")
        self.assertIn("Generated in", status)
        self.assertIn("Steps: 20", status)
        self.assertIn("CFG: 7.0", status)
        self.assertIn("Seed: 1234", status)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_exception(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.side_effect = RuntimeError("CUDA out of memory")
        mock_from_pretrained.return_value = mock_pipe

        app = self._reload_app()

        img, status = app.generate_image("test prompt", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", status)

    def test_mock_mode_generation(self):
        os.environ["MOCK_MODE"] = "1"
        app = self._reload_app()

        self.assertIsNone(app.pipe)
        img, status = app.generate_image("test mock prompt", "", 20, 7.0, 256, 256, 42)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (256, 256))
        self.assertIn("[MOCK MODE]", status)
        self.assertIn("Seed: 42", status)


if __name__ == "__main__":
    unittest.main()
