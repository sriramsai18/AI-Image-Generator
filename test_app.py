import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_import_and_generate_image(self, mock_cuda, mock_from_pretrained):
        # Mock pipeline instance
        mock_pipe_instance = MagicMock()
        mock_from_pretrained.return_value = mock_pipe_instance
        mock_pipe_instance.to.return_value = mock_pipe_instance

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512))]
        mock_pipe_instance.return_value = mock_result

        import app

        # Test empty prompt error handling
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

        # Test valid image generation call
        img, info = app.generate_image("a cyberpunk city", "blurry", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

        # Verify demo blocks initialization
        self.assertIsNotNone(app.demo)

if __name__ == "__main__":
    unittest.main()
