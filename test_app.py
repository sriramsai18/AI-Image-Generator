import unittest
from unittest.mock import MagicMock, patch
import sys
import torch

class TestAppOptimization(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_optimization_and_generation(self, mock_cuda, mock_from_pretrained):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_pipe.unet = mock_unet

        # Setup mock result
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Import app module
        if "app" in sys.modules:
            import importlib
            import app
            importlib.reload(app)
        else:
            import app

        # Assert memory format was set to channels_last on unet
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)

        # Test generate_image function
        img, info = app.generate_image("a cyberpunk city", "", 20, 7.5, 512, 512, 42)
        self.assertEqual(img, mock_image)
        self.assertIn("Generated in", info)

        # Test empty prompt handling
        img_empty, info_empty = app.generate_image("   ", "", 20, 7.5, 512, 512, 42)
        self.assertIsNone(img_empty)
        self.assertIn("Please enter a prompt first", info_empty)

if __name__ == "__main__":
    unittest.main()
