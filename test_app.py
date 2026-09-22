import unittest
from unittest.mock import MagicMock, patch

# Mock StableDiffusionPipeline before app is imported
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
    import app


class TestAppFocusAndAccessibility(unittest.TestCase):
    def test_css_contains_focus_visible_indicators(self):
        self.assertIsNotNone(app)
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)
        self.assertIn(".gradio-container textarea:focus-visible", app.css)
        self.assertIn(".gradio-container a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important", app.css)

    def test_empty_prompt_validation(self):
        image, status = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)


if __name__ == "__main__":
    unittest.main()
