import unittest
from unittest.mock import MagicMock, patch
import sys
import PIL.Image

class TestApp(unittest.TestCase):
    def setUp(self):
        # Clear app module if present to ensure clean initialization with patches
        if "app" in sys.modules:
            del sys.modules["app"]

    @patch("gradio.networking.url_ok", return_value=True)
    @patch("torch.cuda.is_available", return_value=False)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success(self, mock_from_pretrained, mock_cuda, mock_url_ok):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        # Create dummy result image
        dummy_img = PIL.Image.new("RGB", (512, 512))
        mock_result = MagicMock()
        mock_result.images = [dummy_img]
        mock_pipe.return_value = mock_result

        import app

        image, info = app.generate_image("a cute cat", "blurry", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)
        mock_pipe.assert_called()

    @patch("gradio.networking.url_ok", return_value=True)
    @patch("torch.cuda.is_available", return_value=False)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained, mock_cuda, mock_url_ok):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        image, info = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)

if __name__ == "__main__":
    unittest.main()
