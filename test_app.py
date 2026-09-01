import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe
        cls.mock_result = MagicMock()
        cls.mock_result.images = [Image.new("RGB", (512, 512))]
        cls.mock_pipe.return_value = cls.mock_result

        cls.from_pretrained_patcher = patch(
            "diffusers.StableDiffusionPipeline.from_pretrained",
            return_value=cls.mock_pipe,
        )
        cls.cuda_patcher = patch("torch.cuda.is_available", return_value=False)
        cls.from_pretrained_patcher.start()
        cls.cuda_patcher.start()

        import app

        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.from_pretrained_patcher.stop()
        cls.cuda_patcher.stop()

    def test_unet_channels_last(self):
        self.mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt", info)

    def test_generate_image_valid_prompt(self):
        img, info = self.app.generate_image(
            "a beautiful sunset", "", 25, 7.5, 512, 512, 42
        )
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.mock_pipe.assert_called()


if __name__ == "__main__":
    unittest.main()
