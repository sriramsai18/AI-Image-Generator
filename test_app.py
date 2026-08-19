import unittest
from unittest.mock import MagicMock, patch

# Mock heavy imports before app import
with patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_from_pretrained, \
     patch("torch.cuda.is_available", return_value=False):
    mock_pipe = MagicMock()
    mock_from_pretrained.return_value = mock_pipe
    import app

class TestApp(unittest.TestCase):
    def test_css_focus_visible_styles(self):
        """Verify that high-contrast focus-visible styles are present in the CSS."""
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important", app.css)

    def test_generate_image_empty_prompt(self):
        """Verify prompt validation error on empty input."""
        img, info = app.generate_image("", "negative", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    def test_generate_image_whitespace_prompt(self):
        """Verify prompt validation error on whitespace input."""
        img, info = app.generate_image("   ", "negative", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

if __name__ == "__main__":
    unittest.main()
