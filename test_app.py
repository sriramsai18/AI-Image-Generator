import unittest
from unittest.mock import patch
import os

# Patch os.sysconf fallback for cross-platform compatibility if needed
orig_sysconf = getattr(os, 'sysconf', None)
def mock_sysconf(name):
    if orig_sysconf and name in ('SC_PAGE_SIZE', 'SC_PHYS_PAGES'):
        return orig_sysconf(name)
    return 4096

class TestAppAccessibilityAndLogic(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_css_contains_focus_visible_rules(self, mock_from_pretrained):
        import app
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

    def test_generate_image_empty_prompt_validation(self):
        import app
        image, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first!", info)

    def test_generate_image_whitespace_prompt_validation(self):
        import app
        image, info = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first!", info)

if __name__ == "__main__":
    unittest.main()
