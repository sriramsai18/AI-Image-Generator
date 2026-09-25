import unittest
from unittest.mock import MagicMock, patch


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_pipe_patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained")
        cls.mock_from_pretrained = cls.mock_pipe_patcher.start()

        mock_pipe_obj = MagicMock()
        mock_pipe_obj.to.return_value = mock_pipe_obj
        cls.mock_from_pretrained.return_value = mock_pipe_obj

        # Import app after patching
        import app
        cls.app_module = app

    @classmethod
    def tearDownClass(cls):
        cls.mock_pipe_patcher.stop()

    def test_css_contains_focus_visible_rules(self):
        """Verify that the custom CSS includes high-contrast focus-visible rules."""
        css = self.app_module.css
        self.assertIn(":focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important;", css)
        self.assertIn(".gradio-container button:focus-visible", css)

    def test_generate_image_empty_prompt(self):
        """Verify generate_image returns a warning message when prompt is empty."""
        img, info = self.app_module.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)


if __name__ == "__main__":
    unittest.main()
