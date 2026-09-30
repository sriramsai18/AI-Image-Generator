import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline before app import
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe
        cls.mock_pipe.device = torch.device("cpu")
        cls.mock_pipe.unet = MagicMock()
        cls.mock_pipe.vae = MagicMock()

        # Mock result from pipe call
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), color="blue")]
        cls.mock_pipe.return_value = mock_result

        cls.patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=cls.mock_pipe)
        cls.patcher.start()

        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_app_loaded(self):
        self.assertIsNotNone(self.app)
        self.assertIsNotNone(self.app.demo)

    def test_generate_image_valid_prompt(self):
        img, info = self.app.generate_image(
            prompt="A majestic sunset",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image(
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

    def test_generate_image_error_handling(self):
        self.app.pipe.side_effect = Exception("CUDA out of memory")
        img, info = self.app.generate_image(
            prompt="Error test",
            negative_prompt="",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )
        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", info)
        self.app.pipe.side_effect = None

    def test_inference_mode_and_channels_last(self):
        # Verify UNet and VAE had channels_last called on them during initialization
        self.mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)


if __name__ == "__main__":
    unittest.main()
