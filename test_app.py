import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image

class TestApp(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_app_generation_and_optimizations(self, mock_from_pretrained):
        # Create mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        # Configure pipe return value for pipe.to()
        mock_pipe.to.return_value = mock_pipe

        # Return mock PIL image from pipe execution
        fake_img = Image.new("RGB", (512, 512), color="blue")
        mock_result = MagicMock()
        mock_result.images = [fake_img]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Import app module dynamically
        import app

        # Assert channels_last conversion calls on unet and vae
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

        # Test empty prompt error handling
        img, status = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt", status)

        # Test valid prompt generation
        img, status = app.generate_image("a futuristic city", "ugly", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)
        mock_pipe.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_inference_mode_active_during_generation(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        fake_img = Image.new("RGB", (512, 512), color="red")

        def side_effect(*args, **kwargs):
            # Check torch.is_inference_mode_enabled() during pipeline invocation
            self.assertTrue(torch.is_inference_mode_enabled())
            res = MagicMock()
            res.images = [fake_img]
            return res

        mock_pipe.side_effect = side_effect
        mock_from_pretrained.return_value = mock_pipe

        import importlib
        import app
        importlib.reload(app)

        img, status = app.generate_image("a scenic view", "", 20, 7.0, 512, 512, 100)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)

if __name__ == "__main__":
    unittest.main()
