import unittest
from unittest.mock import MagicMock, patch
import importlib
import torch
from PIL import Image

class TestAppInference(unittest.TestCase):
    def setUp(self):
        self.mock_pipe = MagicMock()
        fake_image = Image.new("RGB", (512, 512), color="red")
        mock_result = MagicMock()
        mock_result.images = [fake_image]
        self.mock_pipe.return_value = mock_result
        self.mock_pipe.to.return_value = self.mock_pipe

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe
        import app
        importlib.reload(app)

        img, info = app.generate_image(
            prompt="A beautiful sunset",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 20", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe
        import app
        importlib.reload(app)

        img, info = app.generate_image(
            prompt="   ",
            negative_prompt="",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )

        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_inference_mode_usage(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_pipe
        import app
        importlib.reload(app)

        grad_enabled_during_call = []

        def side_effect_pipe(*args, **kwargs):
            grad_enabled_during_call.append(torch.is_grad_enabled())
            mock_result = MagicMock()
            mock_result.images = [Image.new("RGB", (512, 512))]
            return mock_result

        app.pipe.side_effect = side_effect_pipe

        img, info = app.generate_image(
            prompt="Test prompt",
            negative_prompt="",
            steps=10,
            guidance=7.5,
            width=512,
            height=512,
            seed=123
        )

        self.assertIsNotNone(img)
        self.assertEqual(len(grad_enabled_during_call), 1)
        self.assertFalse(grad_enabled_during_call[0], "torch.is_grad_enabled() should be False during inference")


if __name__ == "__main__":
    unittest.main()
