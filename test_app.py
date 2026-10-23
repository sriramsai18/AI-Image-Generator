import unittest
from unittest.mock import MagicMock, patch
import os

# Handle os.sysconf mock safely for cross-platform psutil compatibility
orig_sysconf = getattr(os, "sysconf", None)

def sysconf_side_effect(key, *args, **kwargs):
    if key == "SC_PAGE_SIZE":
        return 4096
    elif key == "SC_PHYS_PAGES":
        return 2097152
    elif orig_sysconf is not None:
        return orig_sysconf(key, *args, **kwargs)
    return 1


class TestAppPerformanceOptimizations(unittest.TestCase):

    @patch("gradio.networking.url_ok", return_value=True)
    @patch("os.sysconf", side_effect=sysconf_side_effect, create=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_initialization_and_optimizations(self, mock_cuda, mock_from_pretrained, mock_sysconf, mock_url_ok):
        # Setup mock pipeline and sub-modules
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae
        mock_from_pretrained.return_value = mock_pipe

        import importlib
        import app

        importlib.reload(app)

        # Assert channels_last optimization was applied to unet and vae
        import torch
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch("gradio.networking.url_ok", return_value=True)
    @patch("os.sysconf", side_effect=sysconf_side_effect, create=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_functionality(self, mock_cuda, mock_from_pretrained, mock_sysconf, mock_url_ok):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_result = MagicMock()
        mock_result.images = ["fake_pil_image"]
        mock_pipe.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipe

        import importlib
        import app

        importlib.reload(app)

        # Test empty prompt
        img, status = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

        # Test valid prompt generation
        img, status = app.generate_image("A beautiful sunset", "blurry", 25, 7.5, 512, 512, 42)
        self.assertEqual(img, "fake_pil_image")
        self.assertIn("Generated in", status)


if __name__ == "__main__":
    unittest.main()
