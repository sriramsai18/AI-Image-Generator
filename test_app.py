import unittest
from unittest.mock import MagicMock, patch

# Mock StableDiffusionPipeline before importing app to avoid downloading model weights
with patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_from_pretrained:
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_from_pretrained.return_value = mock_pipe

    import app


class TestGenerateImage(unittest.TestCase):

    @patch("app.pipe")
    def test_empty_prompt(self, mock_pipe):
        image, info = app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)
        mock_pipe.assert_not_called()

    @patch("app.pipe")
    def test_successful_generation(self, mock_pipe):
        mock_output = MagicMock()
        mock_output.images = ["fake_pil_image"]
        mock_pipe.return_value = mock_output

        image, info = app.generate_image("a cute cat", "", 25, 7.5, 512, 512, 42)

        self.assertEqual(image, "fake_pil_image")
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 25", info)
        self.assertIn("CFG: 7.5", info)
        self.assertIn("Seed: 42", info)
        mock_pipe.assert_called_once()

    @patch("app.pipe")
    def test_generation_error_handling(self, mock_pipe):
        mock_pipe.side_effect = RuntimeError("CUDA out of memory")

        image, info = app.generate_image("a galaxy", "", 20, 7.5, 512, 512, -1)

        self.assertIsNone(image)
        self.assertIn("Error: CUDA out of memory", info)


if __name__ == "__main__":
    unittest.main()
