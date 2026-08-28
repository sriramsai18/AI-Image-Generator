import importlib
import unittest
from unittest.mock import MagicMock, patch

import PIL.Image
import torch


class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_app_generation(self, mock_from_pretrained):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe

        # Mock UNet and VAE
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae

        # Mock image output
        fake_img = PIL.Image.new("RGB", (512, 512), color="blue")
        mock_result = MagicMock()
        mock_result.images = [fake_img]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Dynamically import/reload app module to trigger model loading with mock
        import app
        importlib.reload(app)

        # Test empty prompt validation
        img, info = app.generate_image("", "ugly", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

        # Test valid image generation
        img, info = app.generate_image("a cyberpunk city", "ugly", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

        # Check that memory_format=torch.channels_last was called on unet & vae
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)

        # Check that pipe was called with expected arguments
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "a cyberpunk city")
        self.assertEqual(kwargs["num_inference_steps"], 20)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_app_exception_handling(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.side_effect = RuntimeError("GPU out of memory")
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        img, info = app.generate_image("a test prompt", "", 10, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("❌ Error: GPU out of memory", info)


if __name__ == "__main__":
    unittest.main()
