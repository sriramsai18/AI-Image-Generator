import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch


class TestAppOptimizations(unittest.TestCase):
    def test_model_loading_optimizations(self):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae

        with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
            import importlib
            import app

            importlib.reload(app)

            # Verify channels_last format was requested for unet and vae
            mock_unet.to.assert_called_with(memory_format=torch.channels_last)
            mock_vae.to.assert_called_with(memory_format=torch.channels_last)

    def test_generate_image_empty_prompt(self):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe

        with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
            import importlib
            import app

            importlib.reload(app)

            img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
            self.assertIsNone(img)
            self.assertIn("Please enter a prompt", info)

    def test_generate_image_success(self):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_image = Image.new("RGB", (512, 512), color="blue")
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
            import importlib
            import app

            importlib.reload(app)

            img, info = app.generate_image(
                "a photo of a cat", "blurry", 20, 7.5, 512, 512, 42
            )
            self.assertIsNotNone(img)
            self.assertIn("Generated in", info)

            # Verify pipeline was called with correct parameters
            mock_pipe.assert_called_once()
            kwargs = mock_pipe.call_args.kwargs
            self.assertEqual(kwargs["prompt"], "a photo of a cat")
            self.assertEqual(kwargs["negative_prompt"], "blurry")
            self.assertEqual(kwargs["num_inference_steps"], 20)
            self.assertEqual(kwargs["guidance_scale"], 7.5)


if __name__ == "__main__":
    unittest.main()
