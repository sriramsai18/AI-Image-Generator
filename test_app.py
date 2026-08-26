import importlib
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestAppPerformance(unittest.TestCase):
    def setUp(self):
        # Setup mocks before importing app
        self.mock_pipe = MagicMock()
        self.mock_pipe.to.return_value = self.mock_pipe
        self.mock_pipe.unet = MagicMock()
        self.mock_pipe.vae = MagicMock()
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        self.mock_pipe.return_value = mock_result

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_model_loading_and_channels_last(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe

        import app
        importlib.reload(app)

        # Check that channels_last memory format was set for unet and vae
        self.mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.inference_mode")
    def test_generate_image_inference_mode(self, mock_inference_mode, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe
        import app
        importlib.reload(app)

        image, info = app.generate_image("a beautiful scenery", "", 25, 7.5, 512, 512, -1)

        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)
        mock_inference_mode.assert_called()


if __name__ == "__main__":
    unittest.main()
