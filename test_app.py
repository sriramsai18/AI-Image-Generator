import importlib
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup mocks before importing app module to avoid downloading heavy weights
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe
        cls.mock_unet = MagicMock()
        cls.mock_vae = MagicMock()
        cls.mock_pipe.unet = cls.mock_unet
        cls.mock_pipe.vae = cls.mock_vae

        dummy_img = Image.new("RGB", (512, 512), color="red")
        mock_output = MagicMock()
        mock_output.images = [dummy_img]
        cls.mock_pipe.return_value = mock_output

        cls.patcher = patch(
            "diffusers.StableDiffusionPipeline.from_pretrained",
            return_value=cls.mock_pipe,
        )
        cls.patcher.start()

        import app

        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        # Ensure app.pipe points to cls.mock_pipe and reset state
        self.app.pipe = self.mock_pipe
        self.mock_pipe.side_effect = None
        dummy_img = Image.new("RGB", (512, 512), color="red")
        mock_output = MagicMock()
        mock_output.images = [dummy_img]
        self.mock_pipe.return_value = mock_output

    def test_app_loaded(self):
        self.assertIsNotNone(self.app)

    def test_generate_image_empty_prompt(self):
        img, status = self.app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_whitespace_prompt(self):
        img, status = self.app.generate_image("   ", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_success(self):
        img, status = self.app.generate_image(
            "a cute cat", "blurry", 25, 7.5, 512, 512, 42
        )
        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)
        self.assertIn("Seed: 42", status)

    @patch("torch.cuda.is_available", return_value=True)
    def test_channels_last_applied_on_cuda(self, mock_cuda):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae

        with patch(
            "diffusers.StableDiffusionPipeline.from_pretrained",
            return_value=mock_pipe,
        ):
            importlib.reload(self.app)
            mock_unet.to.assert_called_with(memory_format=torch.channels_last)
            mock_vae.to.assert_called_with(memory_format=torch.channels_last)

    def test_generate_image_exception(self):
        self.mock_pipe.side_effect = Exception("CUDA error")
        img, status = self.app.generate_image(
            "test prompt", "", 20, 7.0, 512, 512, -1
        )
        self.assertIsNone(img)
        self.assertIn("Error: CUDA error", status)


if __name__ == "__main__":
    unittest.main()
