import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestAppOptimizations(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf")
    def test_channels_last_and_attention_slicing_skip_when_ram_sufficient(
        self, mock_sysconf, mock_cuda_avail, mock_from_pretrained
    ):
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            elif name == "SC_PHYS_PAGES":
                return 2 * 1024 * 1024  # 8 GB
            elif hasattr(os, "sysconf") and isinstance(name, int):
                try:
                    return os.sysconf(name)
                except Exception:
                    pass
            return 0

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        mock_pipe.reset_mock()

        importlib.reload(app)

        import torch

        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf")
    def test_attention_slicing_enabled_when_ram_low(
        self, mock_sysconf, mock_cuda_avail, mock_from_pretrained
    ):
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            elif name == "SC_PHYS_PAGES":
                return 512 * 1024  # 2 GB
            elif hasattr(os, "sysconf") and isinstance(name, int):
                try:
                    return os.sysconf(name)
                except Exception:
                    pass
            return 0

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        mock_pipe.reset_mock()

        importlib.reload(app)

        mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_inference_mode_and_result(
        self, mock_cuda_avail, mock_from_pretrained
    ):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        dummy_img = Image.new("RGB", (512, 512), color="red")
        mock_result = MagicMock()
        mock_result.images = [dummy_img]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        import app
        mock_pipe.reset_mock()

        importlib.reload(app)

        img, info = app.generate_image(
            prompt="a futuristic city",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertEqual(img, dummy_img)
        self.assertIn("Generated in", info)
        mock_pipe.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_empty_prompt(self, mock_cuda_avail, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        mock_pipe.reset_mock()

        importlib.reload(app)

        img, info = app.generate_image(
            prompt="   ",
            negative_prompt="",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)


if __name__ == "__main__":
    unittest.main()
