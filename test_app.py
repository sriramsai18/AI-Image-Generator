import importlib
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestAppPerformance(unittest.TestCase):
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_app_and_generate_image(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        dummy_img = Image.new("RGB", (512, 512), color="red")
        mock_result = MagicMock()
        mock_result.images = [dummy_img]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Check memory layout applied
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

        # Test valid prompt generation
        image, info = app.generate_image(
            prompt="a futuristic city",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )
        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)
        mock_pipe.assert_called_once()

        # Test empty prompt check
        image_empty, info_empty = app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )
        self.assertIsNone(image_empty)
        self.assertIn("Please enter a prompt first", info_empty)


if __name__ == '__main__':
    unittest.main()
