import unittest
from unittest.mock import MagicMock, patch

class TestAppAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline before importing app to avoid loading model weights
        cls.patcher_sd = patch("diffusers.StableDiffusionPipeline.from_pretrained")
        cls.mock_sd = cls.patcher_sd.start()

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        cls.mock_sd.return_value = mock_pipe

        import app
        cls.app_module = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher_sd.stop()

    def test_css_contains_focus_visible_rules(self):
        """Test that custom CSS includes focus-visible rules for key interactive elements."""
        css = self.app_module.css
        self.assertIn(".gradio-container button:focus-visible", css)
        self.assertIn(".gradio-container a:focus-visible", css)
        self.assertIn(".gradio-container input:focus-visible", css)
        self.assertIn(".gradio-container textarea:focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important;", css)

    def test_demo_blocks_initialized(self):
        """Test that the Gradio Blocks demo object is properly instantiated."""
        self.assertIsNotNone(self.app_module.demo)

if __name__ == "__main__":
    unittest.main()
