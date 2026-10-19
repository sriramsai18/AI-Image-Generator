import unittest
import app

class TestAppAccessibilityCSS(unittest.TestCase):
    def test_focus_visible_styles_present(self):
        """Verify that :focus-visible selectors are present in the app's CSS stylesheet."""
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

if __name__ == "__main__":
    unittest.main()
