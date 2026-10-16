import unittest
from unittest.mock import MagicMock, patch
import sys

class TestAppPerformanceOptimizations(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    def test_app_initialization_and_generate_image(self, mock_cuda_is_available, mock_from_pretrained):
        # Setup mocks
        mock_cuda_is_available.return_value = False

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        # Mock result of calling pipe
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        # Import app dynamically
        if "app" in sys.modules:
            import importlib
            app = importlib.reload(sys.modules["app"])
        else:
            import app

        # Assert channels_last format was set on pipe.unet
        mock_pipe.unet.to.assert_called_with(memory_format=app.torch.channels_last)

        # Reference app module to verify initialization side effects
        self.assertIsNotNone(app)

        # Test generate_image with valid prompt
        image, info = app.generate_image(
            prompt="a glowing futuristic city",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        self.assertEqual(image, mock_image)
        self.assertIn("✅ Generated in", info)
        mock_pipe.assert_called_once()

    def test_generate_image_empty_prompt(self):
        import app
        image, info = app.generate_image(
            prompt="   ",
            negative_prompt="",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)

if __name__ == "__main__":
    unittest.main()
