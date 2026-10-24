import unittest
from unittest.mock import MagicMock, patch
from PIL import Image


class TestAppInference(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_empty_prompt(self, mock_cuda, mock_from_pretrained):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        img, status = app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_valid_prompt(self, mock_cuda, mock_from_pretrained):
        # Setup mock pipeline output
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        fake_img = Image.new("RGB", (512, 512))
        mock_output = MagicMock()
        mock_output.images = [fake_img]
        mock_pipe.return_value = mock_output
        mock_from_pretrained.return_value = mock_pipe

        import app
        app.pipe = mock_pipe

        img, status = app.generate_image("a beautiful sunset", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)
        self.assertTrue(mock_pipe.called)


if __name__ == "__main__":
    unittest.main()
