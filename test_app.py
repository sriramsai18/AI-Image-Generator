import unittest
from unittest.mock import patch, MagicMock

# Mock heavy modules before importing app to avoid loading torch/diffusers
with patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_sd, \
     patch("torch.cuda.is_available", return_value=False):

    mock_pipe = MagicMock()
    mock_sd.return_value = mock_pipe
    mock_pipe.to.return_value = mock_pipe

    import app

class TestAppAccessibility(unittest.TestCase):
    def test_css_contains_focus_visible_styles(self):
        """Verify that app.css contains high contrast :focus-visible rules for keyboard accessibility."""
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #39ff14 !important;", app.css)
        self.assertIn("outline-offset: 2px !important;", app.css)

    def test_app_title_and_elements_exist(self):
        """Verify Gradio demo instance is initialized properly."""
        self.assertIsNotNone(app.demo)
        self.assertEqual(app.demo.title, "Text2Image — Sriram")

if __name__ == "__main__":
    unittest.main()
