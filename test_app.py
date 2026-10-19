import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_initialization_and_generation(
        self, mock_cuda, mock_from_pretrained
    ):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_image = Image.new("RGB", (512, 512))
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipe

        import app

        importlib.reload(app)

        self.assertIsNotNone(app.demo)

        # Test empty prompt
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

        # Test successful generation
        img, info = app.generate_image(
            "a beautiful sunset",
            "blurry",
            25,
            7.5,
            512,
            512,
            42,
        )
        self.assertEqual(img, mock_image)
        self.assertIn("Generated in", info)
        mock_pipe.assert_called_once()

    @patch("os.sysconf")
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_low_ram_attention_slicing(
        self, mock_cuda, mock_from_pretrained, mock_sysconf
    ):
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return 524288  # 2 GB total RAM (< 4 GB)
            if hasattr(os, "sysconf"):
                return os.sysconf(name)
            raise ValueError(f"Unknown sysconf key: {name}")

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        importlib.reload(app)

        mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("os.sysconf")
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_sufficient_ram_no_attention_slicing(
        self, mock_cuda, mock_from_pretrained, mock_sysconf
    ):
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return 4194304  # 16 GB total RAM (>= 4 GB)
            if hasattr(os, "sysconf"):
                return os.sysconf(name)
            raise ValueError(f"Unknown sysconf key: {name}")

        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        importlib.reload(app)

        mock_pipe.enable_attention_slicing.assert_not_called()


if __name__ == "__main__":
    unittest.main()
