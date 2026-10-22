import unittest
import app

class TestAppAccessibilityStyles(unittest.TestCase):
    def test_css_contains_focus_visible_rules(self):
        """Verify that app CSS includes high contrast focus-visible accessibility styles."""
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)
        self.assertIn(".gradio-container textarea:focus-visible", app.css)
        self.assertIn(".gradio-container a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)
        self.assertIn("outline-offset: 2px !important;", app.css)

if __name__ == "__main__":
    unittest.main()
