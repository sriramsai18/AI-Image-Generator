import unittest
from unittest.mock import patch, MagicMock
import sys
import torch
import os

# Store original sysconf if it exists to fall back on in side effect mock
original_sysconf = getattr(os, 'sysconf', None)

class TestStableDiffusionOptimization(unittest.TestCase):

    def setUp(self):
        # Prevent Gradio localhost accessibility error crashes
        self.url_ok_patch = patch('gradio.networking.url_ok', return_value=True)
        self.url_ok_patch.start()

        # Clean sys.modules of app so we can reload it and test initialization logic
        if "app" in sys.modules:
            del sys.modules["app"]

    def tearDown(self):
        self.url_ok_patch.stop()

    def mock_sysconf_with_ram(self, ram_bytes):
        page_size = 4096
        pages = ram_bytes // page_size

        def side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return page_size
            elif name == 'SC_PHYS_PAGES':
                return pages
            if original_sysconf:
                try:
                    return original_sysconf(name)
                except ValueError:
                    pass
            raise ValueError(f"sysconf {name} not supported")

        return side_effect

    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('os.sysconf', create=True)
    def test_cpu_init_with_high_ram(self, mock_sysconf, mock_from_pretrained, mock_cuda_avail):
        # Mock high RAM >= 4GB (e.g. 8GB)
        mock_sysconf.side_effect = self.mock_sysconf_with_ram(8 * 1024 * 1024 * 1024)

        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        self.assertIsNotNone(app)

        # Assertions
        mock_from_pretrained.assert_called_once()
        mock_pipe.to.assert_called_with("cpu")

        # Attention slicing should NOT be called (RAM >= 4GB)
        mock_pipe.enable_attention_slicing.assert_not_called()

        # Check channels_last is applied
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('os.sysconf', create=True)
    def test_cpu_init_with_low_ram(self, mock_sysconf, mock_from_pretrained, mock_cuda_avail):
        # Mock low RAM < 4GB (e.g. 2GB)
        mock_sysconf.side_effect = self.mock_sysconf_with_ram(2 * 1024 * 1024 * 1024)

        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        self.assertIsNotNone(app)

        # Attention slicing should be called (RAM < 4GB)
        mock_pipe.enable_attention_slicing.assert_called_once()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_gpu_init_skips_attention_slicing(self, mock_from_pretrained, mock_cuda_avail):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        self.assertIsNotNone(app)

        # Assertions
        mock_from_pretrained.assert_called_once()
        mock_pipe.to.assert_called_with("cuda")

        # Attention slicing should NOT be called on GPU
        mock_pipe.enable_attention_slicing.assert_not_called()

        # Check channels_last is applied
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('os.sysconf', create=True)
    def test_generate_image_inference_mode(self, mock_sysconf, mock_from_pretrained, mock_cuda_avail):
        # Mock high RAM >= 4GB (e.g. 8GB)
        mock_sysconf.side_effect = self.mock_sysconf_with_ram(8 * 1024 * 1024 * 1024)

        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipe

        import app

        # Track if inference mode is enabled when pipeline is called
        def pipe_call_side_effect(*args, **kwargs):
            mock_pipe.inference_mode_active = torch.is_inference_mode_enabled()
            return mock_result
        mock_pipe.side_effect = pipe_call_side_effect

        image, status = app.generate_image(
            prompt="test prompt",
            negative_prompt="test negative",
            steps=10,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        # Verify that pipeline was called with torch.inference_mode active
        self.assertTrue(getattr(mock_pipe, "inference_mode_active", False))
        self.assertEqual(image, mock_image)
        self.assertIn("Generated", status)

if __name__ == "__main__":
    unittest.main()
