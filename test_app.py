import unittest
from unittest.mock import patch, MagicMock
import sys
import torch

# Create mock objects for pipeline and stable diffusion
mock_unet = MagicMock()
mock_vae = MagicMock()

class MockResult:
    def __init__(self):
        self.images = [MagicMock()]

class MockPipeline:
    def __init__(self):
        self.unet = mock_unet
        self.vae = mock_vae
        self.to = MagicMock(return_value=self)
        self.enable_attention_slicing = MagicMock()

    def __call__(self, *args, **kwargs):
        # We can assert whether torch.inference_mode is active
        # torch.is_inference_mode_enabled() returns True if inside the context
        self.inference_mode_active = torch.is_inference_mode_enabled()
        return MockResult()

# Create a patch for the diffusers import and pre-pretrained initialization
# to prevent downloading actual weights during test import
mock_pipe_instance = MockPipeline()

class TestStableDiffusionOptimization(unittest.TestCase):

    def setUp(self):
        # Clear sys.modules of app so we can reload it and test initialization logic under different conditions
        if "app" in sys.modules:
            del sys.modules["app"]
        mock_unet.reset_mock()
        mock_vae.reset_mock()
        mock_pipe_instance.enable_attention_slicing.reset_mock()
        mock_pipe_instance.to.reset_mock()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe_instance)
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_cpu_init_with_high_ram(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # High RAM: e.g. SC_PAGE_SIZE * SC_PHYS_PAGES = 8GB
        mock_sysconf.side_effect = lambda param: 4096 if param == 'SC_PAGE_SIZE' else (2 * 1024**2)

        # Import app to trigger initialization
        import app  # noqa: F401

        # Check channels_last is applied
        mock_unet.to.assert_any_call(memory_format=torch.channels_last)
        mock_vae.to.assert_any_call(memory_format=torch.channels_last)

        # Check enable_attention_slicing was NOT called (RAM >= 4GB)
        mock_pipe_instance.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe_instance)
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_cpu_init_with_low_ram(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # Low RAM: e.g. 2GB
        mock_sysconf.side_effect = lambda param: 4096 if param == 'SC_PAGE_SIZE' else (512 * 1024)

        import app  # noqa: F401

        # Check enable_attention_slicing WAS called (RAM < 4GB)
        mock_pipe_instance.enable_attention_slicing.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe_instance)
    @patch("torch.cuda.is_available", return_value=True)
    def test_gpu_init_skips_attention_slicing(self, mock_cuda, mock_from_pretrained):
        import app  # noqa: F401

        # Check channels_last is applied
        mock_unet.to.assert_any_call(memory_format=torch.channels_last)
        mock_vae.to.assert_any_call(memory_format=torch.channels_last)

        # On GPU, attention slicing should not be called
        mock_pipe_instance.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe_instance)
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_generate_image_inference_mode(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        mock_sysconf.side_effect = lambda param: 4096 if param == 'SC_PAGE_SIZE' else (2 * 1024**2)

        import app

        # Generate image
        img, info = app.generate_image("a prompt", "", 10, 7.5, 512, 512, -1)

        # Assert that pipeline was called with torch.inference_mode active
        self.assertTrue(getattr(mock_pipe_instance, "inference_mode_active", False))
        self.assertIsNotNone(img)
        self.assertIn("Generated", info)

if __name__ == "__main__":
    unittest.main()
