import unittest
from unittest.mock import MagicMock, patch


class TestAppGeneration(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        import app

        image, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_pipe.unet = mock_unet
        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        # Mock result from pipe call
        mock_result = MagicMock()
        mock_result.images = ["fake_pil_image"]
        mock_pipe.return_value = mock_result

        import app
        app.pipe = mock_pipe

        image, info = app.generate_image("a cyberpunk city", "blurry", 25, 7.5, 512, 512, 42)

        self.assertEqual(image, "fake_pil_image")
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)
        mock_pipe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
