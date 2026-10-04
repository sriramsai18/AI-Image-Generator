import sys
import unittest
from unittest.mock import MagicMock

# Mock modules before app import to avoid loading heavy dependencies or models in unit test
sys.modules["diffusers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["gradio"] = MagicMock()

import app  # noqa: E402


class TestAppAccessibilityAndUI(unittest.TestCase):

    def test_css_contains_focus_visible_styles(self):
        """Ensure custom CSS includes accessibility focus-visible rules."""
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

    def test_generate_image_empty_prompt(self):
        """Ensure empty prompt returns helpful warning."""
        img, msg = app.generate_image("", "neg", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", msg)


if __name__ == "__main__":
    unittest.main()
