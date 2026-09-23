import unittest
from unittest.mock import MagicMock, patch
import torch


class TestAppPerformance(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_success(self, mock_cuda, mock_from_pretrained):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae

        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Import app dynamically to trigger mock setup
        import importlib
        import app

        importlib.reload(app)

        # Ensure unet and vae were configured with channels_last
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)

        # Test generate_image function
        img, info = app.generate_image(
            prompt="a beautiful landscape",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertEqual(img, mock_image)
        self.assertIn("✅ Generated in", info)
        self.assertIn("Steps: 20", info)

        # Verify pipeline call arguments
        mock_pipe.assert_called_once()
        _, kwargs = mock_pipe.call_args
        self.assertEqual(kwargs["prompt"], "a beautiful landscape")
        self.assertEqual(kwargs["negative_prompt"], "blurry")

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_empty_prompt(self, mock_cuda, mock_from_pretrained):
        import app

        img, info = app.generate_image(
            prompt="   ",
            negative_prompt="",
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
