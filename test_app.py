import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import importlib
import torch
from PIL import Image

class TestAppOptimizations(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_app_loading_cpu_high_ram(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        original_sysconf = getattr(os, 'sysconf', None)
        def sysconf_side_effect(key):
            if key == 'SC_PAGE_SIZE':
                return 4096
            elif key == 'SC_PHYS_PAGES':
                return 4 * 1024 * 1024  # 16GB RAM
            if original_sysconf:
                try:
                    return original_sysconf(key)
                except Exception:
                    pass
            return 100

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            import app
            importlib.reload(app)
        else:
            import app

        mock_pipe.to.assert_called_with("cpu")
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_app_loading_cpu_low_ram(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        original_sysconf = getattr(os, 'sysconf', None)
        def sysconf_side_effect(key):
            if key == 'SC_PAGE_SIZE':
                return 4096
            elif key == 'SC_PHYS_PAGES':
                return 512 * 1024  # 2GB RAM
            if original_sysconf:
                try:
                    return original_sysconf(key)
                except Exception:
                    pass
            return 100

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            import app
            importlib.reload(app)
        else:
            import app

        mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image(self, mock_cuda, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        inference_mode_active = False

        def pipe_call_side_effect(*args, **kwargs):
            nonlocal inference_mode_active
            inference_mode_active = torch.is_inference_mode_enabled()
            mock_result = MagicMock()
            mock_result.images = [Image.new("RGB", (512, 512))]
            return mock_result

        mock_pipe.side_effect = pipe_call_side_effect
        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            import app
            importlib.reload(app)
        else:
            import app

        img, info = app.generate_image("a photo of a cat", "", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertTrue(inference_mode_active, "torch.inference_mode() should be active during pipe execution")

        img_empty, err = app.generate_image("  ", "", 20, 7.5, 512, 512, 42)
        self.assertIsNone(img_empty)
        self.assertIn("Please enter a prompt", err)

if __name__ == "__main__":
    unittest.main()
