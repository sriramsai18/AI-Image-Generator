import unittest
from unittest.mock import patch, MagicMock

# Mock torch and diffusers before importing app to prevent heavy model loads in unit tests
with patch("torch.cuda.is_available", return_value=False), \
     patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_from_pretrained:

    mock_pipe = MagicMock()
    mock_from_pretrained.return_value = mock_pipe

    import app

class TestAppAccessibility(unittest.TestCase):
    def test_css_contains_focus_visible_rules(self):
        """Verify that high-contrast focus-visible styles are present in app CSS."""
        self.assertIn("button:focus-visible", app.css)
        self.assertIn("input:focus-visible", app.css)
        self.assertIn("textarea:focus-visible", app.css)
        self.assertIn("a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

    def test_generate_image_empty_prompt(self):
        """Verify that empty prompt returns warning message."""
        image, status = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

if __name__ == "__main__":
    unittest.main()
