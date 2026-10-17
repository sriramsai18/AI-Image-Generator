import unittest
from unittest.mock import MagicMock, patch

class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet.to.return_value = mock_pipe.unet
        mock_pipe.vae.to.return_value = mock_pipe.vae

        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_pipe.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipe

        import app

        # Test valid prompt generation
        image, info = app.generate_image("a fantasy castle", "", 25, 7.5, 512, 512, 123)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)

        # Test empty prompt error handling
        img, err = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", err)

if __name__ == "__main__":
    unittest.main()
