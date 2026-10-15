import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

# Save original os.sysconf before patching
REAL_SYSCONF = getattr(os, "sysconf", None)


class TestApp(unittest.TestCase):
    def setUp(self):
        # Create mocks for pipeline and components
        self.mock_pipe = MagicMock()
        self.mock_pipe.to.return_value = self.mock_pipe

        mock_image = Image.new("RGB", (512, 512))
        self.mock_result = MagicMock()
        self.mock_result.images = [mock_image]

        # Record whether inference_mode was enabled when pipe was called
        self.inference_mode_enabled_during_call = False

        def pipe_side_effect(*args, **kwargs):
            self.inference_mode_enabled_during_call = torch.is_inference_mode_enabled()
            return self.mock_result

        self.mock_pipe.side_effect = pipe_side_effect

        self.mock_unet = MagicMock()
        self.mock_vae = MagicMock()
        self.mock_pipe.unet = self.mock_unet
        self.mock_pipe.vae = self.mock_vae

        # Setup patches for pipeline creation and CUDA availability
        self.from_pretrained_patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=self.mock_pipe)
        self.cuda_patcher = patch("torch.cuda.is_available", return_value=False)

        self.mock_from_pretrained = self.from_pretrained_patcher.start()
        self.mock_cuda = self.cuda_patcher.start()

        # Import or reload app so app.pipe uses current self.mock_pipe
        import app
        importlib.reload(app)
        self.app = app

    def tearDown(self):
        self.from_pretrained_patcher.stop()
        self.cuda_patcher.stop()

    def test_generate_image_empty_prompt(self):
        img, status = self.app.generate_image("", "negative", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_success(self):
        img, status = self.app.generate_image("a beautiful landscape", "", 20, 7.0, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)
        self.assertTrue(self.inference_mode_enabled_during_call, "torch.inference_mode should be enabled during pipeline execution")

    def test_generate_image_error_handling(self):
        self.mock_pipe.side_effect = Exception("CUDA OOM or general failure")

        img, status = self.app.generate_image("test prompt", "", 20, 7.0, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA OOM or general failure", status)

    @patch("os.sysconf", create=True)
    def test_app_initialization_optimizations_high_ram(self, mock_sysconf):
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return 2097152  # 8 GB RAM
            if REAL_SYSCONF:
                return REAL_SYSCONF(name)
            raise ValueError(f"Unknown sysconf: {name}")

        mock_sysconf.side_effect = sysconf_side_effect

        import app
        importlib.reload(app)

        # Verify channels_last was configured on unet and vae
        self.mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_vae.to.assert_called_with(memory_format=torch.channels_last)

        # Verify enable_attention_slicing was NOT called since RAM >= 4GB
        self.mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("os.sysconf", create=True)
    def test_app_initialization_optimizations_low_ram(self, mock_sysconf):
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return 524288  # 2 GB RAM
            if REAL_SYSCONF:
                return REAL_SYSCONF(name)
            raise ValueError(f"Unknown sysconf: {name}")

        mock_sysconf.side_effect = sysconf_side_effect

        import app
        importlib.reload(app)

        # Verify enable_attention_slicing WAS called since RAM < 4GB
        self.mock_pipe.enable_attention_slicing.assert_called_once()


if __name__ == "__main__":
    unittest.main()
