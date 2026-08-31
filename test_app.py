import unittest
import app

class TestAccessibilityUX(unittest.TestCase):
    def test_css_focus_visible_indicators(self):
        """Verify that high-contrast focus-visible styles are included in app CSS."""
        css = app.css
        self.assertIn(".gradio-container button:focus-visible", css)
        self.assertIn(".gradio-container input:focus-visible", css)
        self.assertIn(".gradio-container textarea:focus-visible", css)
        self.assertIn(".gradio-container a:focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important;", css)

if __name__ == "__main__":
    unittest.main()
