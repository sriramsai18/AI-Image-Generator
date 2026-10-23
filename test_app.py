import unittest
from app import css

class TestAppUXCSS(unittest.TestCase):
    def test_focus_visible_styles_exist(self):
        """Verify high-contrast focus-visible CSS rules exist for keyboard accessibility."""
        self.assertIn(":focus-visible", css)
        self.assertIn(".gradio-container button:focus-visible", css)
        self.assertIn(".gradio-container input:focus-visible", css)
        self.assertIn(".gradio-container textarea:focus-visible", css)
        self.assertIn(".gradio-container a:focus-visible", css)

if __name__ == "__main__":
    unittest.main()
