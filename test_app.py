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
        self.call_mock = MagicMock()
        fake_res = MagicMock()
        fake_res.images = [Image.new("RGB", (512, 512))]
        self.call_mock.return_value = fake_res

    def to(self, device):
        return self

    def enable_attention_slicing(self):
        pass

    def __call__(self, **kwargs):
        return self.call_mock(**kwargs)

mock_pipe = DummyPipeline()

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
    import app

class TestApp(unittest.TestCase):
    def setUp(self):
        fake_res = MagicMock()
        fake_res.images = [Image.new("RGB", (512, 512))]
        mock_pipe.call_mock.reset_mock()
        mock_pipe.call_mock.side_effect = None
        mock_pipe.call_mock.return_value = fake_res

    def test_pipeline_optimization(self):
        # Verify that channels_last memory format was applied to UNet and VAE
        self.assertTrue(mock_pipe.unet.conv.weight.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(mock_pipe.vae.conv.weight.is_contiguous(memory_format=torch.channels_last))

    def test_generate_image_empty_prompt(self):
        image, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt", info)

    def test_generate_image_valid(self):
        image, info = app.generate_image("a beautiful scenery", "", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)

    def test_generate_image_inference_mode(self):
        called_in_inference_mode = []
        fake_res = MagicMock()
        fake_res.images = [Image.new("RGB", (512, 512))]

        def fake_call(**kwargs):
            called_in_inference_mode.append(torch.is_inference_mode_enabled())
            return fake_res

        mock_pipe.call_mock.side_effect = fake_call
        image, info = app.generate_image("cyberpunk car", "", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)
        self.assertTrue(len(called_in_inference_mode) > 0)
        self.assertTrue(called_in_inference_mode[0], "Pipeline call was not executed inside torch.inference_mode")

    def test_generate_image_exception_handling(self):
        mock_pipe.call_mock.side_effect = RuntimeError("CUDA out of memory")
        image, info = app.generate_image("crash test", "", 10, 7.0, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("CUDA out of memory", info)

if __name__ == "__main__":
    unittest.main()
