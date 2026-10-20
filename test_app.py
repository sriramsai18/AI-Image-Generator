import unittest
from unittest.mock import MagicMock, patch
import sys


class TestAppPerformanceAndFunctionality(unittest.TestCase):
    def setUp(self):
        self.mock_pipe = MagicMock()
        self.mock_result = MagicMock()
        self.mock_result.images = ["fake_image_object"]
        self.mock_pipe.return_value = self.mock_result
        self.mock_pipe.to.return_value = self.mock_pipe

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_valid_prompt(self, mock_cuda, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe

        if "app" in sys.modules:
            import importlib
            import app
            importlib.reload(app)
        else:
            import app

        image, info = app.generate_image(
            prompt="a cute kitten",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertEqual(image, "fake_image_object")
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 20", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_empty_prompt(self, mock_cuda, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe

        if "app" in sys.modules:
            import importlib
            import app
            importlib.reload(app)
        else:
            import app

        image, info = app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )

        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_inference_mode_and_channels_last(self, mock_cuda, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe

        if "app" in sys.modules:
            import importlib
            import app
            importlib.reload(app)
        else:
            import app

        import torch

        # Verify channels_last was called on unet and vae
        self.mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

        # Verify torch.is_inference_mode_enabled() inside pipe invocation
        def check_inference_mode(*args, **kwargs):
            self.assertTrue(torch.is_inference_mode_enabled())
            return self.mock_result

        self.mock_pipe.side_effect = check_inference_mode

        app.generate_image(
            prompt="cyberpunk city",
            negative_prompt="",
            steps=10,
            guidance=7.0,
            width=512,
            height=512,
            seed=-1,
        )

        self.mock_pipe.assert_called()


if __name__ == "__main__":
    unittest.main()
