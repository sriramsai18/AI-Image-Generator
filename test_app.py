import importlib
import unittest
from unittest.mock import MagicMock, patch
import torch


class TestApp(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_pipeline_initialization_and_channels_last(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        self.assertIsNotNone(app)
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        image, status = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        image, status = app.generate_image(
            "a beautiful sunset",
            "blurry",
            25,
            7.5,
            512,
            512,
            12345
        )

        self.assertEqual(image, mock_image)
        self.assertIn("Generated in", status)
        self.assertIn("Steps: 25", status)
        mock_pipe.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_exception_handling(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.side_effect = RuntimeError("Out of memory")
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        image, status = app.generate_image("prompt", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Error: Out of memory", status)


if __name__ == "__main__":
    unittest.main()
