import unittest
from unittest.mock import patch, MagicMock

# Mock dependencies before importing app module
mock_gr = MagicMock()
mock_torch = MagicMock()
mock_diffusers = MagicMock()

with patch.dict('sys.modules', {
    'gradio': mock_gr,
    'torch': mock_torch,
    'diffusers': mock_diffusers,
}):
    import app

class TestAppAccessibility(unittest.TestCase):
    def test_css_contains_focus_visible_styles(self):
        """Test that custom CSS includes focus-visible rules for keyboard accessibility."""
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)
        self.assertIn(".gradio-container textarea:focus-visible", app.css)
        self.assertIn(".gradio-container a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

    def test_generate_image_empty_prompt(self):
        """Test prompt validation for empty strings."""
        result, message = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(result)
        self.assertIn("Please enter a prompt first", message)

if __name__ == '__main__':
    unittest.main()
