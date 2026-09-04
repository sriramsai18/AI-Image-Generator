import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch


class TestAppInference(unittest.TestCase):

    def setUp(self):
        # Create mock pipeline
        self.mock_pipe = MagicMock()
        self.mock_image = Image.new("RGB", (512, 512), color="blue")
        mock_result = MagicMock()
        mock_result.images = [self.mock_image]
        self.mock_pipe.return_value = mock_result

        # Configure pipe.to returning itself
        self.mock_pipe.to.return_value = self.mock_pipe
        self.mock_pipe.unet = MagicMock()
        self.mock_pipe.vae = MagicMock()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe
        import app

        img, msg = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", msg)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_valid_prompt(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe
        import app
        app.pipe = self.mock_pipe

        img, msg = app.generate_image("a beautiful sunset", "blurry", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", msg)
        self.assertIn("Steps: 20", msg)
        self.assertIn("Seed: 42", msg)
        self.mock_pipe.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_exception(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe
        import app
        app.pipe = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        img, msg = app.generate_image("a cat", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", msg)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_memory_format_optimization(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe
        import importlib
        import app

        importlib.reload(app)
        self.mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)


if __name__ == "__main__":
    unittest.main()
