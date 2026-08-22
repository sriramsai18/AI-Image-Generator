import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import importlib

orig_sysconf = os.sysconf

class TestAppPerformanceOptimizations(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf")
    def test_attention_slicing_enabled_on_low_ram(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # 2GB RAM
        def sysconf_side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return 4096
            if name == 'SC_PHYS_PAGES':
                return 524288  # 2 GB
            return orig_sysconf(name)

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            app = importlib.reload(sys.modules["app"])
        else:
            import app

        self.assertIsNotNone(app)
        mock_pipe.enable_attention_slicing.assert_called_once()
        mock_pipe.unet.to.assert_called_with(memory_format=importlib.import_module("torch").channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=importlib.import_module("torch").channels_last)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf")
    def test_attention_slicing_disabled_on_sufficient_ram(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # 8GB RAM
        def sysconf_side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return 4096
            if name == 'SC_PHYS_PAGES':
                return 2097152  # 8 GB
            return orig_sysconf(name)

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            app = importlib.reload(sys.modules["app"])
        else:
            import app

        self.assertIsNotNone(app)
        mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_inference_mode(self, mock_cuda, mock_from_pretrained):
        import torch

        inference_mode_active = False

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe

        def mock_pipe_call(*args, **kwargs):
            nonlocal inference_mode_active
            inference_mode_active = torch.is_inference_mode_enabled()
            mock_res = MagicMock()
            mock_res.images = ["fake_pil_image"]
            return mock_res

        mock_pipe.side_effect = mock_pipe_call
        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            app = importlib.reload(sys.modules["app"])
        else:
            import app

        img, info = app.generate_image("a cyberpunk city", "", 20, 7.5, 512, 512, 42)

        self.assertTrue(inference_mode_active, "torch.inference_mode should be active during pipe invocation")
        self.assertEqual(img, "fake_pil_image")
        self.assertIn("✅ Generated in", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_empty_prompt(self, mock_cuda, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            app = importlib.reload(sys.modules["app"])
        else:
            import app

        img, info = app.generate_image("   ", "", 20, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", info)

if __name__ == "__main__":
    unittest.main()
