import unittest
from unittest.mock import MagicMock, patch

class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import importlib
        import app
        importlib.reload(app)

        img, info = app.generate_image(
            prompt="a fantasy landscape",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertEqual(img, mock_image)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import importlib
        import app
        importlib.reload(app)

        img, info = app.generate_image(
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
