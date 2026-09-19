import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestAppOptimizations(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    def test_model_loading_channels_last(self, mock_cuda_available, mock_from_pretrained):
        mock_cuda_available.return_value = False
        mock_pipe = MagicMock()
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    def test_attention_slicing_skipped_on_sufficient_ram(
        self, mock_cuda_available, mock_from_pretrained
    ):
        mock_cuda_available.return_value = False
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        orig_sysconf = getattr(os, "sysconf", None)

        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return 4 * 1024 * 1024  # 16 GB RAM
            if orig_sysconf is not None:
                return orig_sysconf(name)
            raise ValueError(f"Invalid sysconf key: {name}")

        with patch("os.sysconf", side_effect=sysconf_side_effect, create=True):
            import app
            importlib.reload(app)

            mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    def test_attention_slicing_enabled_on_low_ram(
        self, mock_cuda_available, mock_from_pretrained
    ):
        mock_cuda_available.return_value = False
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        orig_sysconf = getattr(os, "sysconf", None)

        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return 512 * 1024  # 2 GB RAM
            if orig_sysconf is not None:
                return orig_sysconf(name)
            raise ValueError(f"Invalid sysconf key: {name}")

        with patch("os.sysconf", side_effect=sysconf_side_effect, create=True):
            import app
            mock_pipe.enable_attention_slicing.reset_mock()
            importlib.reload(app)

            mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("app.pipe")
    def test_generate_image_inference_mode(self, mock_pipe):
        fake_result = MagicMock()
        fake_image = MagicMock()
        fake_result.images = [fake_image]
        mock_pipe.return_value = fake_result

        import app

        is_inference_active = None

        def pipe_side_effect(*args, **kwargs):
            nonlocal is_inference_active
            is_inference_active = torch.is_inference_mode_enabled()
            return fake_result

        mock_pipe.side_effect = pipe_side_effect

        img, status = app.generate_image(
            prompt="A fast running horse",
            negative_prompt="",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertEqual(img, fake_image)
        self.assertIn("Generated in", status)
        self.assertTrue(is_inference_active)


if __name__ == "__main__":
    unittest.main()
