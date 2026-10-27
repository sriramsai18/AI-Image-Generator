import unittest
from unittest.mock import MagicMock, patch
import sys


class TestAppPerformanceOptimizations(unittest.TestCase):

    def setUp(self):
        # Patch StableDiffusionPipeline.from_pretrained so importing app.py doesn't download model weights
        self.patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained")
        self.mock_from_pretrained = self.patcher.start()

        self.mock_pipe = MagicMock()
        self.mock_pipe.to.return_value = self.mock_pipe
        self.mock_pipe.unet = MagicMock()
        self.mock_pipe.vae = MagicMock()

        mock_result = MagicMock()
        mock_result.images = ["fake_image_object"]
        self.mock_pipe.return_value = mock_result

        self.mock_from_pretrained.return_value = self.mock_pipe

        if "app" in sys.modules:
            del sys.modules["app"]

    def tearDown(self):
        self.patcher.stop()

    def test_pipeline_memory_format_and_inference_mode(self):
        import app

        import torch
        self.mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

        img, info = app.generate_image("a futuristic city", "", 25, 7.5, 512, 512, 42)

        self.assertEqual(img, "fake_image_object")
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

    def test_empty_prompt_validation(self):
        import app
        img, info = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", info)


if __name__ == "__main__":
    unittest.main()
