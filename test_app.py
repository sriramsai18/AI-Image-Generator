import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import torch

# Store original sysconf if it exists to fall back on in side effect mock
original_sysconf = getattr(os, 'sysconf', None)

class TestAppOptimization(unittest.TestCase):
    def setUp(self):
        # Clear module from sys.modules to force a full reload and rerun initialization code on import
        if 'app' in sys.modules:
            del sys.modules['app']

    def mock_sysconf_with_ram(self, ram_bytes):
        page_size = 4096
        pages = ram_bytes // page_size

        def side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return page_size
            elif name == 'SC_PHYS_PAGES':
                return pages
            if original_sysconf:
                return original_sysconf(name)
            raise ValueError(f"sysconf {name} not supported")

        return side_effect

    @patch('gradio.networking.url_ok', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('os.sysconf', create=True)
    def test_cpu_high_ram_no_attention_slicing(self, mock_sysconf, mock_from_pretrained, mock_cuda_avail, mock_url_ok):
        # Mock high RAM >= 4GB (e.g., 8GB)
        mock_sysconf.side_effect = self.mock_sysconf_with_ram(8 * 1024 * 1024 * 1024)

        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        # Load the app module (this executes the module-level initialization once)
        import app
        self.assertIsNotNone(app)

        # Verify from_pretrained and to were called correctly
        mock_from_pretrained.assert_called_once()
        mock_pipe.to.assert_called_with("cpu")

        # Attention slicing should NOT be enabled because RAM >= 4GB
        mock_pipe.enable_attention_slicing.assert_not_called()

        # Verify channels_last optimization is applied to unet and vae
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch('gradio.networking.url_ok', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('os.sysconf', create=True)
    def test_cpu_low_ram_enables_attention_slicing(self, mock_sysconf, mock_from_pretrained, mock_cuda_avail, mock_url_ok):
        # Mock low RAM < 4GB (e.g., 2GB)
        mock_sysconf.side_effect = self.mock_sysconf_with_ram(2 * 1024 * 1024 * 1024)

        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        # Load the app module
        import app
        self.assertIsNotNone(app)

        # Verify from_pretrained and to were called correctly
        mock_from_pretrained.assert_called_once()
        mock_pipe.to.assert_called_with("cpu")

        # Attention slicing SHOULD be enabled because RAM < 4GB
        mock_pipe.enable_attention_slicing.assert_called_once()

        # Verify channels_last optimization is applied to unet and vae
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch('gradio.networking.url_ok', return_value=True)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_cuda_available_no_attention_slicing(self, mock_from_pretrained, mock_cuda_avail, mock_url_ok):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        # Load the app module
        import app
        self.assertIsNotNone(app)

        # Verify from_pretrained and to were called correctly
        mock_from_pretrained.assert_called_once()
        mock_pipe.to.assert_called_with("cuda")

        # Attention slicing should NOT be enabled on CUDA
        mock_pipe.enable_attention_slicing.assert_not_called()

        # Verify channels_last optimization is applied to unet and vae
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch('gradio.networking.url_ok', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('os.sysconf', create=True)
    @patch('torch.inference_mode')
    def test_generate_image_context_managers(self, mock_inf_mode, mock_sysconf, mock_from_pretrained, mock_cuda_avail, mock_url_ok):
        # Mock high RAM >= 4GB (e.g., 8GB)
        mock_sysconf.side_effect = self.mock_sysconf_with_ram(8 * 1024 * 1024 * 1024)

        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_result = MagicMock()
        mock_result.images = ["mock_pil_image"]
        mock_pipe.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipe

        # Setup mock_inf_mode as context manager
        mock_context = MagicMock()
        mock_inf_mode.return_value = mock_context

        # Load the app module
        import app

        image, info = app.generate_image(
            prompt="test prompt",
            negative_prompt="test negative",
            steps=10,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        # Verify the returned image and info message
        self.assertEqual(image, "mock_pil_image")
        self.assertIn("✅ Generated in", info)

        # Verify inference_mode context manager was entered and exited correctly
        mock_inf_mode.assert_called_once()
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()

    @patch('gradio.networking.url_ok', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('os.sysconf', create=True)
    def test_generate_image_empty_prompt(self, mock_sysconf, mock_from_pretrained, mock_cuda_avail, mock_url_ok):
        mock_sysconf.side_effect = self.mock_sysconf_with_ram(8 * 1024 * 1024 * 1024)
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        image, info = app.generate_image(
            prompt="   ",
            negative_prompt="",
            steps=10,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )
        self.assertIsNone(image)
        self.assertIn("⚠️ Please enter a prompt first!", info)

    @patch('gradio.networking.url_ok', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('os.sysconf', create=True)
    def test_generate_image_exception(self, mock_sysconf, mock_from_pretrained, mock_cuda_avail, mock_url_ok):
        mock_sysconf.side_effect = self.mock_sysconf_with_ram(8 * 1024 * 1024 * 1024)
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.side_effect = Exception("Inference failed")
        mock_from_pretrained.return_value = mock_pipe

        import app

        image, info = app.generate_image(
            prompt="test prompt",
            negative_prompt="",
            steps=10,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )
        self.assertIsNone(image)
        self.assertIn("❌ Error: Inference failed", info)

if __name__ == "__main__":
    unittest.main()
