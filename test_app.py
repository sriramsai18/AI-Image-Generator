import unittest
from unittest.mock import MagicMock, patch
from PIL import Image


class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        # Mock pipeline setup
        mock_pipe = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app

        img, info = app.generate_image("", "negative", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success(self, mock_from_pretrained):
        # Mock pipeline output
        mock_pipe = MagicMock()
        mock_image = Image.new("RGB", (512, 512))
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipe

        import app
        app.pipe = mock_pipe

        img, info = app.generate_image("a cute cat", "", 20, 7.5, 512, 512, 42)

        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args[1]
        self.assertEqual(kwargs["prompt"], "a cute cat")
        self.assertIsNone(kwargs["negative_prompt"])
        self.assertEqual(kwargs["num_inference_steps"], 20)
        self.assertEqual(kwargs["guidance_scale"], 7.5)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_exception_handling(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.side_effect = RuntimeError("CUDA out of memory")
        mock_from_pretrained.return_value = mock_pipe

        import app
        app.pipe = mock_pipe

        img, info = app.generate_image("a galaxy", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", info)


if __name__ == "__main__":
    unittest.main()
