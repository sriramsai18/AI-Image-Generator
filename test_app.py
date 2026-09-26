import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestAppOptimizations(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    def test_generate_image_uses_inference_mode_and_returns_result(self, mock_cuda, mock_from_pretrained):
        mock_cuda.return_value = False
        mock_pipe = MagicMock()
        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Mock result from pipeline call
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        image, info = app.generate_image(
            prompt="a beautiful fantasy landscape",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        self.assertEqual(image, mock_image)
        self.assertIn("✅ Generated in", info)
        mock_pipe.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    def test_generate_image_empty_prompt(self, mock_cuda, mock_from_pretrained):
        mock_cuda.return_value = False
        mock_pipe = MagicMock()
        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        import app
        importlib.reload(app)

        image, info = app.generate_image(
            prompt="   ",
            negative_prompt="",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )

        self.assertIsNone(image)
        self.assertIn("⚠️ Please enter a prompt first!", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    def test_cuda_channels_last_applied(self, mock_cuda, mock_from_pretrained):
        mock_cuda.return_value = True
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae
        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        import app
        importlib.reload(app)

        import torch
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)

if __name__ == "__main__":
    unittest.main()
