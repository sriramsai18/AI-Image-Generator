import unittest
from unittest.mock import MagicMock, patch

import torch


class TestAppGeneration(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def setUp(self, mock_from_pretrained):
        self.mock_pipe = MagicMock()
        self.mock_pipe.to.return_value = self.mock_pipe
        self.mock_pipe.unet = MagicMock()
        self.mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = self.mock_pipe

        # Import app after mocking model loading
        import app
        self.app = app
        self.app.pipe = self.mock_pipe

    def test_empty_prompt(self):
        img, info = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    def test_generate_image_success(self):
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        self.mock_pipe.return_value = mock_result

        img, info = self.app.generate_image("a cute cat", "", 20, 7.5, 512, 512, 42)

        self.assertEqual(img, mock_image)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)
        self.mock_pipe.assert_called_once()

    def test_generate_image_error_handling(self):
        self.mock_pipe.side_effect = RuntimeError("CUDA out of memory")

        img, info = self.app.generate_image("test prompt", "", 20, 7.5, 512, 512, -1)

        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", info)
        self.mock_pipe.side_effect = None

    def test_inference_mode_active(self):
        # Verify torch.is_inference_mode_enabled() during pipeline execution
        was_inference_mode = []

        def side_effect(*args, **kwargs):
            was_inference_mode.append(torch.is_inference_mode_enabled())
            mock_result = MagicMock()
            mock_result.images = [MagicMock()]
            return mock_result

        self.mock_pipe.side_effect = side_effect

        img, _info = self.app.generate_image("a futuristic city", "", 10, 7.5, 256, 256, -1)

        self.assertIsNotNone(img)
        self.assertTrue(len(was_inference_mode) > 0)
        self.assertTrue(was_inference_mode[0], "pipe() should execute within torch.inference_mode()")
        self.mock_pipe.side_effect = None


if __name__ == "__main__":
    unittest.main()
