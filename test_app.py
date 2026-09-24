import unittest
from unittest.mock import MagicMock, patch

import torch


class TestAppOptimizations(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_model_loading_and_memory_format(self, mock_from_pretrained):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import importlib

        import app
        importlib.reload(app)

        # Verify channels_last memory format was applied to UNet and VAE
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_inference_mode(self, mock_from_pretrained):
        inference_mode_during_pipe = None

        def side_effect_pipe(*args, **kwargs):
            nonlocal inference_mode_during_pipe
            inference_mode_during_pipe = torch.is_inference_mode_enabled()
            mock_res = MagicMock()
            mock_res.images = ["fake_image"]
            return mock_res

        mock_pipe = MagicMock()
        mock_pipe.side_effect = side_effect_pipe
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import importlib

        import app
        importlib.reload(app)

        img, status = app.generate_image("a test prompt", "", 20, 7.5, 512, 512, 42)

        self.assertTrue(inference_mode_during_pipe, "torch.inference_mode() should be active during pipe execution")
        self.assertEqual(img, "fake_image")
        self.assertIn("Generated in", status)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import importlib

        import app
        importlib.reload(app)

        img, status = app.generate_image("", "", 20, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

if __name__ == "__main__":
    unittest.main()
