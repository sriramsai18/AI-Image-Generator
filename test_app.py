import unittest
from unittest.mock import MagicMock, patch

class TestAppUI(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_css_focus_visible(self, mock_cuda, mock_from_pretrained):
        # Mock pipeline to avoid downloading weights
        mock_pipe = MagicMock()
        mock_from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        import app

        self.assertIsNotNone(app.demo)
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important", app.css)

if __name__ == "__main__":
    unittest.main()
