import unittest
from unittest.mock import MagicMock, patch

class TestAppAccessibility(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_css_contains_focus_visible_styles(self, mock_cuda, mock_sd_pipe):
        mock_pipe = MagicMock()
        mock_sd_pipe.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        import app
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)
        self.assertIn(".gradio-container textarea:focus-visible", app.css)
        self.assertIn(".gradio-container a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

if __name__ == "__main__":
    unittest.main()
