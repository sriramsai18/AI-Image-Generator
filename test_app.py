import unittest
from unittest.mock import MagicMock, patch
import importlib
import torch

class TestAppOptimization(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_pipeline_optimization_initialization(self, mock_from_pretrained):
        # Setup mock pipeline with mock unet and vae
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae
        mock_from_pretrained.return_value = mock_pipe

        # Reload app module to execute initialization logic with mocks
        import app
        importlib.reload(app)

        # Verify channels_last format was applied
        mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=torch.channels_last)
        self.assertIsNotNone(app)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        import app
        img, status = app.generate_image("   ", "neg", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_result = MagicMock()
        mock_result.images = ["fake_image_object"]
        mock_pipe.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        img, status = app.generate_image("a cyberpunk cat", "blurry", 20, 7.0, 512, 512, 42)
        self.assertEqual(img, "fake_image_object")
        self.assertIn("Generated in", status)
        self.assertIn("Steps: 20", status)
        self.assertIn("CFG: 7.0", status)
        self.assertIn("Seed: 42", status)

        # Verify pipe call arguments
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "a cyberpunk cat")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 20)
        self.assertEqual(kwargs["guidance_scale"], 7.0)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)
        self.assertIsNotNone(kwargs["generator"])

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_exception(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.side_effect = Exception("CUDA out of memory")
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        img, status = app.generate_image("test prompt", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", status)


if __name__ == "__main__":
    unittest.main()
