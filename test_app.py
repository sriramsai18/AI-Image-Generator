import unittest
from unittest.mock import MagicMock, patch


class TestAppAccessibility(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_demo_and_css_focus_visible(self, mock_cuda, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app

        self.assertIsNotNone(app.demo)
        self.assertIn(":focus-visible", app.css)
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)


if __name__ == "__main__":
    unittest.main()
