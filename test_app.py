import unittest
from unittest.mock import MagicMock, patch
import torch


class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        import app

        image, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first!", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success_and_inference_mode(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        inference_mode_active = False

        def mock_pipe_call(*args, **kwargs):
            nonlocal inference_mode_active
            inference_mode_active = torch.is_inference_mode_enabled()
            mock_result = MagicMock()
            mock_result.images = ["fake_image"]
            return mock_result

        mock_pipe.side_effect = mock_pipe_call

        import app
        app.pipe = mock_pipe

        image, info = app.generate_image("a beautiful sunset", "", 25, 7.5, 512, 512, -1)
        self.assertEqual(image, "fake_image")
        self.assertIn("Generated in", info)
        self.assertTrue(inference_mode_active, "torch.inference_mode() should be enabled during pipeline execution")

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_exception_handling(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.side_effect = Exception("CUDA out of memory")

        import app
        app.pipe = mock_pipe

        image, info = app.generate_image("a beautiful sunset", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Error: CUDA out of memory", info)


if __name__ == "__main__":
    unittest.main()
