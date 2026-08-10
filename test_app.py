import unittest
from unittest.mock import patch, MagicMock
import importlib
import torch
import os

class TestApp(unittest.TestCase):

    def setUp(self):
        # Prevent Gradio localhost accessibility error crashes
        self.url_ok_patch = patch('gradio.networking.url_ok', return_value=True)
        self.url_ok_patch.start()

    def tearDown(self):
        self.url_ok_patch.stop()

    @patch('torch.cuda.is_available', return_value=False)
    @patch('os.sysconf', create=True)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_cpu_high_ram_no_attention_slicing(self, mock_from_pretrained, mock_sysconf, mock_cuda_available):
        # Set RAM to 8GB (8 * 1024 * 1024 * 1024)
        def sysconf_side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return 4096
            elif name == 'SC_PHYS_PAGES':
                return 2097152
            raise ValueError()
        mock_sysconf.side_effect = sysconf_side_effect

        # Mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Assertions
        mock_pipe.enable_attention_slicing.assert_not_called()
        mock_pipe.unet.to.assert_any_call(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_any_call(memory_format=torch.channels_last)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('os.sysconf', create=True)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_cpu_low_ram_with_attention_slicing(self, mock_from_pretrained, mock_sysconf, mock_cuda_available):
        # Set RAM to 2GB (2 * 1024 * 1024 * 1024)
        def sysconf_side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return 4096
            elif name == 'SC_PHYS_PAGES':
                return 524288
            raise ValueError()
        mock_sysconf.side_effect = sysconf_side_effect

        # Mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Assertions
        mock_pipe.enable_attention_slicing.assert_called_once()
        mock_pipe.unet.to.assert_any_call(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_any_call(memory_format=torch.channels_last)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_cuda_available_no_attention_slicing(self, mock_from_pretrained, mock_cuda_available):
        # Mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Assertions
        mock_pipe.enable_attention_slicing.assert_not_called()
        mock_pipe.unet.to.assert_any_call(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_any_call(memory_format=torch.channels_last)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_generate_image_inference_mode(self, mock_from_pretrained, mock_cuda_available):
        # Mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        # Configure return value of pipeline call
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Check inference mode wrapping
        image, status = app.generate_image(
            prompt="test prompt",
            negative_prompt="test negative",
            steps=10,
            guidance=7.5,
            width=256,
            height=256,
            seed=42
        )
        self.assertEqual(image, mock_image)
        self.assertIn("Generated", status)

if __name__ == '__main__':
    unittest.main()
