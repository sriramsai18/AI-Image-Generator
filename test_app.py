import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Mock Gradio before importing app to avoid launching webserver or rendering templates
sys.modules["gradio"] = MagicMock()


class TestAppOptimizations(unittest.TestCase):
    def setUp(self):
        # Explicitly ensure clean environment
        os.environ["MOCK_MODE"] = "0"
        if "app" in sys.modules:
            del sys.modules["app"]

    def tearDown(self):
        if "app" in sys.modules:
            del sys.modules["app"]
        if "MOCK_MODE" in os.environ:
            del os.environ["MOCK_MODE"]

    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_attention_slicing_under_low_ram(self, mock_sysconf, mock_cuda):
        # 2GB RAM: SC_PAGE_SIZE = 4096, SC_PHYS_PAGES = 524288
        mock_sysconf.side_effect = lambda key: 4096 if "PAGE_SIZE" in key else 524288

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
            import app
            self.assertIsNotNone(app)
            # Memory format channels_last check
            mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
            mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)
            # Enable attention slicing called under low RAM
            mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_attention_slicing_under_high_ram(self, mock_sysconf, mock_cuda):
        # 8GB RAM: SC_PAGE_SIZE = 4096, SC_PHYS_PAGES = 2097152
        mock_sysconf.side_effect = lambda key: 4096 if "PAGE_SIZE" in key else 2097152

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
            import app
            self.assertIsNotNone(app)
            # Memory format channels_last check
            mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
            mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)
            # Enable attention slicing should NOT be called under high RAM
            mock_pipe.enable_attention_slicing.assert_not_called()

    def test_mock_mode_generation(self):
        os.environ["MOCK_MODE"] = "1"
        import app
        self.assertIsNotNone(app)

        image, info = app.generate_image("Test Prompt", "", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 25", info)


if __name__ == "__main__":
    unittest.main()
