import unittest
from unittest.mock import MagicMock, patch

class TestAppAccessibility(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_css_contains_focus_visible(self, mock_cuda, mock_from_pretrained):
        # Mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important", app.css)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_empty_prompt(self, mock_cuda, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

if __name__ == "__main__":
    unittest.main()
