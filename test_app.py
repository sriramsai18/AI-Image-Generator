import unittest
from unittest.mock import patch, MagicMock

# Mock StableDiffusionPipeline before importing app to avoid downloading model weights
with patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_pretrained:
    mock_pipe = MagicMock()
    mock_pretrained.return_value = mock_pipe
    import app

class TestAppAccessibility(unittest.TestCase):

    def test_css_contains_focus_visible(self):
        """Verify high contrast :focus-visible rules exist in custom CSS."""
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #39ff14 !important;", app.css)

    def test_generate_image_validation(self):
        """Verify empty prompt returns error message without running pipeline."""
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

if __name__ == "__main__":
    unittest.main()
