import importlib
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestAppPerformanceOptimizations(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def setUp(self, mock_from_pretrained):
        # Set up mock pipeline
        self.mock_pipe = MagicMock()
        self.mock_pipe.to.return_value = self.mock_pipe
        self.mock_pipe.unet = MagicMock()
        self.mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = self.mock_pipe

        # Import or reload app
        import app
        self.app = importlib.reload(app)

    def test_app_loaded(self):
        self.assertIsNotNone(self.app)

    def test_channels_last_optimization_applied(self):
        # Verify unet.to and vae.to were called with memory_format=torch.channels_last
        self.mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    def test_generate_image_success_with_inference_mode(self):
        mock_img = Image.new("RGB", (512, 512))
        mock_result = MagicMock()
        mock_result.images = [mock_img]

        inference_mode_active = []

        def side_effect(*args, **kwargs):
            inference_mode_active.append(torch.is_inference_mode_enabled())
            return mock_result

        self.mock_pipe.side_effect = side_effect

        img, info = self.app.generate_image(
            prompt="a beautiful futuristic city",
            negative_prompt="blurry",
            steps=20,
            guidance=7.0,
            width=512,
            height=512,
            seed=42,
        )

        self.assertEqual(img, mock_img)
        self.assertIn("✅ Generated in", info)
        self.assertIn("Steps: 20", info)
        self.assertIn("CFG: 7.0", info)
        self.assertIn("Seed: 42", info)

        # Verify pipe call arguments
        self.mock_pipe.assert_called_once()
        _, kwargs = self.mock_pipe.call_args
        self.assertEqual(kwargs["prompt"], "a beautiful futuristic city")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 20)
        self.assertEqual(kwargs["guidance_scale"], 7.0)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)
        self.assertIsNotNone(kwargs["generator"])

        # Check that torch.is_inference_mode_enabled() was True when pipe was called
        self.assertEqual(len(inference_mode_active), 1)
        self.assertTrue(inference_mode_active[0])

    def test_generate_image_error_handling(self):
        self.mock_pipe.side_effect = RuntimeError("Pipeline error")
        img, info = self.app.generate_image("prompt", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("❌ Error: Pipeline error", info)


if __name__ == "__main__":
    unittest.main()
