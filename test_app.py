import unittest
from unittest.mock import patch, MagicMock

# Mock diffusers and torch before importing app to avoid downloading heavy models or CUDA checks
with patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_sd, \
     patch("torch.cuda.is_available", return_value=False):
    mock_pipe = MagicMock()
    mock_sd.return_value = mock_pipe
    mock_pipe.to.return_value = mock_pipe

    import app

class TestAppAccessibility(unittest.TestCase):
    def test_css_focus_visible_rules_present(self):
        """Verify that high-contrast focus-visible accessibility rules are present in CSS."""
        self.assertIn("button:focus-visible", app.css)
        self.assertIn("input:focus-visible", app.css)
        self.assertIn("textarea:focus-visible", app.css)
        self.assertIn("a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

    def test_generate_image_empty_prompt(self):
        """Verify that empty prompt returns warning message."""
        img, msg = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", msg)

if __name__ == "__main__":
    unittest.main()
