import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock diffusers before importing app to prevent downloading models during unit tests
sys.modules['diffusers'] = MagicMock()

class TestAppAccessibility(unittest.TestCase):
    @patch("gradio.networking.url_ok", return_value=True)
    def test_css_focus_visible_styles_present(self, mock_url_ok):
        import app
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)
        self.assertIn("box-shadow: 0 0 14px rgba(230,57,70,0.6) !important;", app.css)

    def test_demo_blocks_initialized(self):
        import app
        self.assertIsNotNone(app.demo)
        self.assertEqual(app.demo.title, "Text2Image — Sriram")

if __name__ == "__main__":
    unittest.main()
