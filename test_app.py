import unittest
from unittest.mock import patch, MagicMock
import os

# Patch sysconf if needed
original_sysconf = getattr(os, 'sysconf', None)

def mock_sysconf(name):
    if original_sysconf is not None:
        try:
            return original_sysconf(name)
        except Exception:
            pass
    if name == 'SC_PAGE_SIZE':
        return 4096
    if name == 'SC_PHYS_PAGES':
        return 1024 * 1024
    return 0

class TestApp(unittest.TestCase):

    @classmethod
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('torch.cuda.is_available', return_value=False)
    @patch('os.sysconf', side_effect=mock_sysconf, create=True)
    def setUpClass(cls, mock_sys, mock_cuda, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        cls.app = app

    def test_css_contains_focus_visible(self):
        self.assertIn(':focus-visible', self.app.css)
        self.assertIn('outline: 2px solid #e63946 !important;', self.app.css)

    def test_empty_prompt_returns_warning(self):
        img, msg = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", msg)


if __name__ == '__main__':
    unittest.main()
