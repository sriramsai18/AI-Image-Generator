import importlib
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch


class TestAppOptimizations(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_model_channels_last_applied(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae

        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        import app

        importlib.reload(app)

        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)

    def test_generate_image_success(self):
        import app

        mock_pipe = MagicMock()
        mock_image = Image.new("RGB", (512, 512), color="blue")
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        with patch.object(app, "pipe", mock_pipe):
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
        mock_pipe.assert_called_once()

    def test_generate_image_empty_prompt(self):
        import app

        img, info = app.generate_image(
            prompt="   ",
            negative_prompt="",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )

        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    def test_generate_image_error_handling(self):
        import app

        mock_pipe = MagicMock()
        mock_pipe.side_effect = RuntimeError("CUDA out of memory")

        with patch.object(app, "pipe", mock_pipe):
            img, info = app.generate_image(
                prompt="A test prompt",
                negative_prompt="",
                steps=25,
                guidance=7.5,
                width=512,
                height=512,
                seed=-1,
            )

        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", info)


if __name__ == "__main__":
    unittest.main()
