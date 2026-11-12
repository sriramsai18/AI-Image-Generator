import os
import unittest
from unittest.mock import MagicMock
from PIL import Image

# Enable mock mode for testing without downloading real models
os.environ["MOCK_MODE"] = "1"
import app  # noqa: E402


class TestAppPerformance(unittest.TestCase):
    def setUp(self):
        self.mock_pipe = MagicMock()
        mock_image = Image.new("RGB", (64, 64), color="blue")
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        self.mock_pipe.return_value = mock_result
        self.mock_pipe.to.return_value = self.mock_pipe
        app.pipe = self.mock_pipe

    def test_empty_prompt_returns_warning(self):
        image, status = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)
        self.mock_pipe.assert_not_called()

    def test_generate_image_success(self):
        image, status = app.generate_image("a cute cat", "", 20, 7.0, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", status)
        self.mock_pipe.assert_called_once()
        _, kwargs = self.mock_pipe.call_args
        self.assertEqual(kwargs["prompt"], "a cute cat")
        self.assertEqual(kwargs["num_inference_steps"], 20)
        self.assertEqual(kwargs["guidance_scale"], 7.0)

    def test_generate_image_runs_with_inference_mode(self):
        in_inference_mode = False

        def mock_call(*args, **kwargs):
            nonlocal in_inference_mode
            in_inference_mode = torch.is_inference_mode_enabled()
            mock_res = MagicMock()
            mock_res.images = [Image.new("RGB", (64, 64), color="red")]
            return mock_res

        import torch
        self.mock_pipe.side_effect = mock_call

        image, status = app.generate_image("test prompt", "", 20, 7.0, 512, 512, -1)
        self.assertIsNotNone(image)
        self.assertTrue(in_inference_mode, "Inference must be executed inside torch.inference_mode()")


if __name__ == "__main__":
    unittest.main()
