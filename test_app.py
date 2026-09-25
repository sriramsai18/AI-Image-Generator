import os
import unittest
from unittest.mock import MagicMock, patch

import torch

# Handle psutil / os.sysconf
orig_sysconf = getattr(os, 'sysconf', None)
def mock_sysconf(name):
    if orig_sysconf and name in os.sysconf_names:
        return orig_sysconf(name)
    if name == 'SC_PAGE_SIZE':
        return 4096
    if name == 'SC_PHYS_PAGES':
        return 2097152
    return 0

class TestAppOptimization(unittest.TestCase):

    @patch('os.sysconf', side_effect=mock_sysconf, create=True)
    @patch('gradio.networking.url_ok', return_value=True)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_generate_image_and_optimizations(self, mock_from_pretrained, mock_url_ok, mock_sys):
        mock_pipe = MagicMock()
        mock_result = MagicMock()
        mock_result.images = ["mock_pil_image"]
        mock_pipe.return_value = mock_result
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_pipe.unet.to.return_value = mock_pipe.unet
        mock_pipe.vae.to.return_value = mock_pipe.vae
        mock_from_pretrained.return_value = mock_pipe

        import app

        # Check prompt validation
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt", info)

        # Check successful image generation
        img, info = app.generate_image("a cyberpunk runner", "", 25, 7.5, 512, 512, 1234)
        self.assertEqual(img, "mock_pil_image")
        self.assertIn("Generated in", info)

        # Verify unet & vae memory_format channels_last conversion calls
        mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

if __name__ == '__main__':
    unittest.main()
