import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_pipe.to.return_value = mock_pipe

        mock_image = Image.new("RGB", (512, 512), color="red")
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        import app

        # Test empty prompt handling
        img, info = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

        # Test valid prompt generation
        img, info = app.generate_image("a beautiful sunset", "blurry", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

        # Test random seed (-1)
        img, info = app.generate_image("cyberpunk city", "", 20, 7.0, 512, 512, -1)
        self.assertIsNotNone(img)
        self.assertIn("Seed: random", info)


if __name__ == "__main__":
    unittest.main()
