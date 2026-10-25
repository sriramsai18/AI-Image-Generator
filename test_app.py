import unittest
from unittest.mock import MagicMock, patch
import torch
import importlib

class TestAppPerformance(unittest.TestCase):
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_pipeline_memory_format_and_inference_mode(self, mock_from_pretrained):
        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Verify channels_last memory format optimization was called on unet and vae
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)

        # Verify generate_image wraps pipeline call in torch.inference_mode()
        with patch('torch.inference_mode') as mock_inference_mode:
            mock_pipe.return_value.images = [MagicMock()]
            img, info = app.generate_image("a futuristic city", "", 25, 7.5, 512, 512, -1)
            mock_inference_mode.assert_called_once()
            self.assertIsNotNone(img)
            self.assertIn("Generated in", info)

if __name__ == '__main__':
    unittest.main()
