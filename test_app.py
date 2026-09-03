import unittest
from unittest.mock import MagicMock, patch
import PIL.Image

class TestApp(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_valid_prompt(self, mock_from_pretrained):
        # Mock pipeline
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_image = PIL.Image.new("RGB", (512, 512))
        mock_output.images = [mock_image]
        mock_pipe.return_value = mock_output
        mock_pipe.to.return_value = mock_pipe

        mock_from_pretrained.return_value = mock_pipe

        # Import app with patched pipeline
        import app

        image, info = app.generate_image(
            prompt="a cute cat",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertIsNotNone(image)
        self.assertIn("✅ Generated in", info)
        self.assertIn("Seed: 42", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        import app

        image, info = app.generate_image(
            prompt="   ",
            negative_prompt="",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )

        self.assertIsNone(image)
        self.assertIn("⚠️ Please enter a prompt first!", info)

if __name__ == "__main__":
    unittest.main()
