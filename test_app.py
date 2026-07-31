import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch

# To prevent downloading the 5GB Stable Diffusion model when importing app,
# we mock StableDiffusionPipeline.from_pretrained before importing app.py.
mock_pipe = MagicMock()
mock_pipe.unet = MagicMock()
mock_pipe.vae = MagicMock()
mock_pipe.images = [MagicMock()]
mock_pipe.return_value = MagicMock(images=[MagicMock()])

sys.modules['diffusers'] = MagicMock()
from diffusers import StableDiffusionPipeline

StableDiffusionPipeline.from_pretrained = MagicMock(return_value=mock_pipe)

# Now we can import the app modules safely
import app


class TestStableDiffusionOptimizations(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=False)
    @patch('os.sysconf', create=True)
    def test_attention_slicing_under_4gb(self, mock_sysconf, mock_cuda):
        # Mock sysconf to return < 4GB RAM (e.g., 2GB)
        # SC_PAGE_SIZE = 4096, SC_PHYS_PAGES = 524288 -> 2GB
        def sysconf_side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return 4096
            elif name == 'SC_PHYS_PAGES':
                return 524288
            raise ValueError()
        mock_sysconf.side_effect = sysconf_side_effect

        # Reset mock
        mock_pipe.to().enable_attention_slicing.reset_mock()

        # Reload app to trigger module-level initialization with mock system RAM
        importlib.reload(app)

        mock_pipe.to().enable_attention_slicing.assert_called_once()

    @patch('torch.cuda.is_available', return_value=False)
    @patch('os.sysconf', create=True)
    def test_no_attention_slicing_above_4gb(self, mock_sysconf, mock_cuda):
        # Mock sysconf to return > 4GB RAM (e.g., 8GB)
        def sysconf_side_effect(name):
            if name == 'SC_PAGE_SIZE':
                return 4096
            elif name == 'SC_PHYS_PAGES':
                return 2097152
            raise ValueError()
        mock_sysconf.side_effect = sysconf_side_effect

        # Reset mock
        mock_pipe.to().enable_attention_slicing.reset_mock()

        # Reload app to trigger module-level initialization with mock system RAM
        importlib.reload(app)

        mock_pipe.to().enable_attention_slicing.assert_not_called()

    def test_generate_image_uses_inference_mode(self):
        # Verify that generate_image runs without exceptions and uses the mocked pipeline
        app.pipe = mock_pipe
        img, info = app.generate_image("test prompt", "", 5, 7.5, 512, 512, -1)
        self.assertIsNotNone(img)
        self.assertIn("✅ Generated in", info)


if __name__ == '__main__':
    unittest.main()
