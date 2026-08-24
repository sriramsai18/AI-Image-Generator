import unittest
from unittest.mock import MagicMock, patch

class TestAppUXAndCSS(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("gradio.networking.url_ok", return_value=True)
    def test_css_focus_visible_styles(self, mock_url_ok, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        self.assertIsNotNone(app)

        css = app.css
        self.assertIn(".gradio-container button:focus-visible", css)
        self.assertIn(".gradio-container input:focus-visible", css)
        self.assertIn(".gradio-container textarea:focus-visible", css)
        self.assertIn(".gradio-container a:focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important;", css)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("gradio.networking.url_ok", return_value=True)
    def test_generate_image_empty_prompt_validation(self, mock_url_ok, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        img, info = app.generate_image("", "ugly", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

if __name__ == "__main__":
    unittest.main()
