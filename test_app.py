import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import importlib

# Get the original sysconf function if it exists
original_sysconf = getattr(os, 'sysconf', None)

class TestApp(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf")
    def test_attention_slicing_under_4gb(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # Mock sysconf to simulate 2GB of RAM (2 * 1024 * 1024 * 1024)
        # SC_PAGE_SIZE = 4096, SC_PHYS_PAGES = 524288 -> 2GB
        def sysconf_side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return 4096
            elif name == 'SC_PHYS_PAGES':
                return 524288
            # Fallback to original sysconf for initialization/import-time lookups (like psutil)
            if original_sysconf is not None:
                try:
                    return original_sysconf(name)
                except Exception:
                    pass
            raise ValueError()
        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        # Force reload app.py module to trigger its top-level initialization code
        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])
        else:
            import app
            _ = app

        # Verify channels_last formatting is called on UNet and VAE
        mock_pipe.unet.to.assert_any_call(memory_format=unittest.mock.ANY)
        mock_pipe.vae.to.assert_any_call(memory_format=unittest.mock.ANY)

        # Since RAM < 4GB, enable_attention_slicing should have been called
        mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf")
    def test_no_attention_slicing_over_4gb(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # Mock sysconf to simulate 8GB of RAM
        def sysconf_side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return 4096
            elif name == 'SC_PHYS_PAGES':
                return 2097152
            # Fallback to original sysconf
            if original_sysconf is not None:
                try:
                    return original_sysconf(name)
                except Exception:
                    pass
            raise ValueError()
        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])
        else:
            import app
            _ = app

        # Since RAM >= 4GB, enable_attention_slicing should NOT be called
        mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_generate_image_inference_mode(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # Simulating environment where sysconf fails for memory size checks,
        # but returns valid values for initialization lookups (like psutil).
        def sysconf_side_effect(name):
            if name in ('SC_PAGE_SIZE', 'SC_PHYS_PAGES'):
                raise AttributeError()
            if original_sysconf is not None:
                return original_sysconf(name)
            raise AttributeError()
        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()

        # Mock the __call__ method of the pipeline to return an object with list of images
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])
        else:
            import app
            _ = app

        # Test generate_image
        image, info = sys.modules["app"].generate_image("a beautiful sunrise", "", 10, 7.5, 512, 512, 42)

        self.assertEqual(image, mock_image)
        self.assertIn("✅ Generated in", info)
        mock_pipe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
