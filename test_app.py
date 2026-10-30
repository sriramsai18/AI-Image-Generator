import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup mocks before importing app module
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

    def test_channels_last_applied(self):
        self.mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_vae.to.assert_called_with(memory_format=torch.channels_last)

    def test_generate_image_exception(self):
        self.mock_pipe.side_effect = Exception("CUDA error")
        img, status = self.app.generate_image(
            "test prompt", "", 20, 7.0, 512, 512, -1
        )
        self.assertIsNone(img)
        self.assertIn("Error: CUDA error", status)
        self.mock_pipe.side_effect = None  # Reset side effect


if __name__ == "__main__":
    unittest.main()
