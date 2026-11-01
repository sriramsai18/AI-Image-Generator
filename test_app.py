import unittest
from unittest.mock import MagicMock, patch


class TestAppAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_sd, \
             patch("torch.cuda.is_available", return_value=False):
            mock_pipe = MagicMock()
            mock_pipe.to.return_value = mock_pipe
            mock_sd.return_value = mock_pipe
            import app
            cls.app = app

    def test_focus_visible_css_defined(self):
        """Verify high-contrast focus-visible styles exist in the app CSS."""
        self.assertIn(".gradio-container button:focus-visible", self.app.css)
        self.assertIn(".gradio-container a:focus-visible", self.app.css)
        self.assertIn(".gradio-container input:focus-visible", self.app.css)
        self.assertIn(".gradio-container textarea:focus-visible", self.app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", self.app.css)
        self.assertIn("outline-offset: 2px !important;", self.app.css)
        self.assertIn("box-shadow: 0 0 12px rgba(230,57,70,0.6) !important;", self.app.css)

    def test_app_components_exist(self):
        """Verify core UI blocks components are configured."""
        self.assertIsNotNone(self.app.demo)
        self.assertIsNotNone(self.app.prompt)
        self.assertIsNotNone(self.app.negative_prompt)
        self.assertIsNotNone(self.app.generate_btn)
        self.assertIsNotNone(self.app.output_image)
        self.assertIsNotNone(self.app.info_text)


if __name__ == "__main__":
    unittest.main()
