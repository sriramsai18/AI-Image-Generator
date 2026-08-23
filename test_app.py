import unittest
from unittest.mock import MagicMock, patch

class TestAppAccessibilityCSS(unittest.TestCase):
    def test_css_contains_focus_visible_rules(self):
        """Test that app.py CSS contains high-contrast focus-visible selectors."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("button:focus-visible", content)
        self.assertIn("input:focus-visible", content)
        self.assertIn("textarea:focus-visible", content)
        self.assertIn("a:focus-visible", content)
        self.assertIn("[role=\"button\"]:focus-visible", content)
        self.assertIn("outline: 2px solid #e63946 !important;", content)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_imports_and_launches_blocks(self, mock_cuda, mock_sd):
        """Test that app module loads and instantiates Gradio Blocks without error."""
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_sd.return_value = mock_pipe

        import app
        self.assertIsNotNone(app.demo)
        self.assertIn(":focus-visible", app.css)

if __name__ == "__main__":
    unittest.main()
