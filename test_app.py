import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestAppOptimizations(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_and_optimizations(self, mock_cuda, mock_from_pretrained):
        # Create a mock pipeline with unet and vae
        mock_pipe = MagicMock()
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_image = Image.new("RGB", (512, 512), color="blue")
        mock_pipe_result = MagicMock()
        mock_pipe_result.images = [mock_image]
        mock_pipe.return_value = mock_pipe_result
        mock_from_pretrained.return_value = mock_pipe

        # Import app after mocks are set up
        import app

        # Verify channels_last memory format was applied to unet and vae
        mock_pipe.unet.to.assert_called_with(memory_format=app.torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=app.torch.channels_last)

        # Test generate_image function
        img, info = app.generate_image(
            prompt="A beautiful sunset",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

        # Test empty prompt check
        empty_img, empty_info = app.generate_image("", "", 20, 7.5, 512, 512, -1)
        self.assertIsNone(empty_img)
        self.assertIn("Please enter a prompt", empty_info)


if __name__ == "__main__":
    unittest.main()
