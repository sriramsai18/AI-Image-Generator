import unittest
from unittest.mock import MagicMock, patch
import importlib

# Mock StableDiffusionPipeline before app is loaded to avoid loading actual weights in tests
mock_pipe_obj = MagicMock()
mock_pipe_obj.to.return_value = mock_pipe_obj

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe_obj):
    import app

class TestAppOptimizations(unittest.TestCase):

    def test_generate_image_empty_prompt(self):
        img, status = app.generate_image("  ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    @patch.object(app, "pipe")
    def test_generate_image_success(self, mock_pipe):
        from PIL import Image
        dummy_img = Image.new("RGB", (512, 512), color="red")
        mock_output = MagicMock()
        mock_output.images = [dummy_img]
        mock_pipe.return_value = mock_output

        img, status = app.generate_image("a futuristic city", "blurry", 25, 7.5, 512, 512, 123)
        self.assertEqual(img, dummy_img)
        self.assertIn("Generated in", status)
        mock_pipe.assert_called_once()

    @patch.object(app, "pipe")
    def test_generate_image_handles_exception(self, mock_pipe):
        mock_pipe.side_effect = Exception("CUDA out of memory")

        img, status = app.generate_image("a cat", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", status)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_channels_last_applied_on_cuda(self, mock_from_pretrained, mock_cuda):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        importlib.reload(app)

        mock_pipe.unet.to.assert_called_with(memory_format=app.torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=app.torch.channels_last)


if __name__ == "__main__":
    unittest.main()
