import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class DummyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)

class DummyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)

class DummyPipeline:
    def __init__(self):
        self.unet = DummyUNet()
        self.vae = DummyVAE()

    def to(self, device):
        return self

    def enable_attention_slicing(self):
        pass

    def __call__(self, **kwargs):
        res = MagicMock()
        res.images = [Image.new("RGB", (512, 512))]
        return res

mock_pipe = DummyPipeline()

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
    import app

class TestApp(unittest.TestCase):
    def test_generate_image_empty_prompt(self):
        image, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt", info)

    def test_generate_image_valid(self):
        image, info = app.generate_image("a beautiful scenery", "", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)

if __name__ == "__main__":
    unittest.main()
