import unittest
from unittest.mock import MagicMock, patch

class TestAppAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_from_pretrained:
            mock_pipe = MagicMock()
            mock_from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            import app
            cls.app_module = app

    def test_css_contains_focus_visible_styles(self):
        css = getattr(self.app_module, "css", "")
        self.assertIn(":focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important;", css)
        self.assertIn("outline-offset: 2px !important;", css)

    def test_app_demo_instance(self):
        self.assertIsNotNone(self.app_module.demo)

if __name__ == "__main__":
    unittest.main()
