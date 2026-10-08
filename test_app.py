import unittest
from unittest.mock import MagicMock, patch


class TestAppAccessibilityAndUI(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_css_contains_focus_visible_styles(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        self.assertIsNotNone(app)
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)
        self.assertIn(".gradio-container button:focus-visible", app.css)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_app_demo_initialized(self, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        self.assertIsNotNone(app.demo)


if __name__ == "__main__":
    unittest.main()
