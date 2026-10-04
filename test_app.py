import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock torch and diffusers before app.py is imported to allow running tests
# without loading heavy weights or requiring GPU / model downloads.

class TestAppOptimization(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_generation_and_optimizations(self, mock_cuda_available, mock_from_pretrained):
        # Create mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe

        # Mock UNet and VAE layers
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae

        # Mock pipeline execution return result
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Import or reload app module
        if "app" in sys.modules:
            import importlib
            import app
            importlib.reload(app)
        else:
            import app

        # Verify channels_last was called on UNet and VAE
        mock_unet.to.assert_called()
        mock_vae.to.assert_called()

        # Test empty prompt validation
        img, status = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt", status)

        # Test image generation
        img, status = app.generate_image("a cyberpunk city", "blurry", 25, 7.5, 512, 512, 42)
        self.assertEqual(img, mock_image)
        self.assertIn("Generated in", status)
        self.assertIn("Seed: 42", status)

if __name__ == "__main__":
    unittest.main()
