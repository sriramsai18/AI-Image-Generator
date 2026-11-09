import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

# Ensure MOCK_MODE is enabled for imports and testing
os.environ["MOCK_MODE"] = "1"
import app


class TestApp(unittest.TestCase):
    def setUp(self):
        # Ensure app reference is ready
        self.assertIsNotNone(app)

    def test_generate_image_empty_prompt(self):
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("⚠️ Please enter a prompt first!", info)

        img, info = app.generate_image("   ", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("⚠️ Please enter a prompt first!", info)

    def test_generate_image_success(self):
        img, info = app.generate_image(
            prompt="A majestic lion on a cliff",
            negative_prompt="blurry",
            steps=20,
            guidance=7.0,
            width=512,
            height=512,
            seed=42,
        )
        self.assertIsNotNone(img)
        self.assertIsInstance(img, Image.Image)
        self.assertIn("✅ Generated in", info)
        self.assertIn("Steps: 20", info)
        self.assertIn("CFG: 7.0", info)
        self.assertIn("Seed: 42", info)

    def test_generate_image_random_seed(self):
        img, info = app.generate_image(
            prompt="Futuristic city",
            negative_prompt="",
            steps=15,
            guidance=8.0,
            width=512,
            height=512,
            seed=-1,
        )
        self.assertIsNotNone(img)
        self.assertIn("Seed: random", info)

    def test_generate_image_handles_exception(self):
        # Temporarily mock pipe to raise an exception
        with patch.object(app, "pipe", side_effect=RuntimeError("CUDA out of memory")):
            img, info = app.generate_image("prompt", "", 20, 7.5, 512, 512, -1)
            self.assertIsNone(img)
            self.assertIn("❌ Error: CUDA out of memory", info)

    def test_channels_last_applied_on_unet_and_vae(self):
        # Mock diffusers StableDiffusionPipeline.from_pretrained
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae

        with (
            patch.dict(os.environ, {"MOCK_MODE": "0"}),
            patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe),
        ):
            importlib.reload(app)
            mock_unet.to.assert_called_with(memory_format=torch.channels_last)
            mock_vae.to.assert_called_with(memory_format=torch.channels_last)

        # Restore MOCK_MODE for subsequent tests
        os.environ["MOCK_MODE"] = "1"
        importlib.reload(app)

    def test_inference_mode_used_during_generation(self):
        called_with_inference_mode = False

        def mock_call(*args, **kwargs):
            nonlocal called_with_inference_mode
            called_with_inference_mode = torch.is_inference_mode_enabled()
            return app.PipelineResult(images=[Image.new("RGB", (64, 64))])

        with patch.object(app, "pipe", side_effect=mock_call):
            app.generate_image("A futuristic robot", "", 20, 7.0, 512, 512, 100)
            self.assertTrue(called_with_inference_mode, "torch.inference_mode() should be enabled during pipeline call")


if __name__ == "__main__":
    unittest.main()
