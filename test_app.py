import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image


class TestAppInference(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_and_optimizations(self, mock_from_pretrained):
        # Create mock pipeline
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae
        mock_pipe.to.return_value = mock_pipe

        # Mock generated image result
        mock_image = Image.new("RGB", (512, 512))
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Import app module dynamically with mocked dependencies
        import app

        # Verify channels_last was called on UNet and VAE
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)

        # Test empty prompt input
        img, status = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt", status)

        # Test valid prompt generation
        img, status = app.generate_image("a cyberpunk city", "blurry", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("✅ Generated in", status)
        self.assertTrue(mock_pipe.called)


if __name__ == "__main__":
    unittest.main()
