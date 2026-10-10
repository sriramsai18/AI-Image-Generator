import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestApp(unittest.TestCase):
    @classmethod
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def setUpClass(cls, mock_from_pretrained):
        # Create dummy pipeline mock
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe
        cls.mock_pipe.unet = MagicMock()
        cls.mock_pipe.vae = MagicMock()

        # Configure pipeline return value for inference
        dummy_img = Image.new("RGB", (512, 512), color="white")
        mock_output = MagicMock()
        mock_output.images = [dummy_img]
        cls.mock_pipe.return_value = mock_output

        mock_from_pretrained.return_value = cls.mock_pipe

        # Import app after mocking from_pretrained
        import app
        cls.app = app

    def test_generate_image_valid_prompt(self):
        img, info = self.app.generate_image(
            prompt="a futuristic city",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)


if __name__ == "__main__":
    unittest.main()
