import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestAppOptimizations(unittest.TestCase):
    def setUp(self):
        # Clear app module if already loaded to force reload
        if "app" in sys.modules:
            del sys.modules["app"]

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    @patch("os.sysconf", create=True)
    def test_cpu_optimization_with_high_ram(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # Setup mocks
        mock_cuda.return_value = False

        # 8 GB of RAM: 8 * 1024**3
        mock_sysconf.side_effect = lambda name: 8 * 1024**3 if name == 'SC_PHYS_PAGES' else (1 if name == 'SC_PAGE_SIZE' else 0)

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        # Verify that enable_attention_slicing was NOT called on high RAM CPU
        mock_pipe.enable_attention_slicing.assert_not_called()

        # Verify channels_last optimization (once implemented)
        mock_pipe.unet.to.assert_any_call(memory_format=importlib.import_module("torch").channels_last)
        mock_pipe.vae.to.assert_any_call(memory_format=importlib.import_module("torch").channels_last)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available")
    @patch("os.sysconf", create=True)
    def test_cpu_optimization_with_low_ram(self, mock_sysconf, mock_cuda, mock_from_pretrained):
        # Setup mocks
        mock_cuda.return_value = False

        # 2 GB of RAM: 2 * 1024**3
        mock_sysconf.side_effect = lambda name: 2 * 1024**3 if name == 'SC_PHYS_PAGES' else (1 if name == 'SC_PAGE_SIZE' else 0)

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        # Verify that enable_attention_slicing WAS called on low RAM CPU
        mock_pipe.enable_attention_slicing.assert_called_once()

if __name__ == "__main__":
    unittest.main()
