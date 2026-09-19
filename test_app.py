import unittest
from unittest.mock import patch, MagicMock
import os

# Mock sysconf side effect to prevent psutil / gradio issues on unhandled keys
orig_sysconf = getattr(os, 'sysconf', None)
def mock_sysconf(name):
    if name == 'SC_PAGE_SIZE':
        return 4096
    if name == 'SC_PHYS_PAGES':
        return 4194304
    if orig_sysconf:
        return orig_sysconf(name)
    return 100

class TestAppAccessibility(unittest.TestCase):
    @patch('os.sysconf', side_effect=mock_sysconf, create=True)
    @patch('gradio.networking.url_ok', return_value=True)
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('torch.cuda.is_available', return_value=False)
    def test_css_contains_focus_visible_styles(self, mock_cuda, mock_sd_pipe, mock_url_ok, mock_sysconf_fn):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_sd_pipe.return_value = mock_pipe

        import app
        self.assertIn("/* Accessibility Focus States */", app.css)
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)
        self.assertIsNotNone(app.demo)

if __name__ == '__main__':
    unittest.main()
