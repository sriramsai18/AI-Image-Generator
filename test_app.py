import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestAppOptimization(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_success(self, mock_cuda_is_avail, mock_from_pretrained):
        # Mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        # Mock generation output
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Import or reload app so mock_pipe is bound to app.pipe
        import app
        importlib.reload(app)

        # Test empty prompt
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt", info)

        # Test valid prompt generation
        img, info = app.generate_image("a futuristic city", "blurry", 25, 7.5, 512, 512, 42)
        self.assertEqual(img, mock_image)
        self.assertIn("Generated in", info)

        # Verify pipe was called
        mock_pipe.assert_called_once()
        _, kwargs = mock_pipe.call_args
        self.assertEqual(kwargs["prompt"], "a futuristic city")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 25)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_error(self, mock_cuda_is_avail, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.side_effect = RuntimeError("CUDA OOM")
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        img, info = app.generate_image("a futuristic city", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA OOM", info)

if __name__ == "__main__":
    unittest.main()
