import unittest
from unittest.mock import MagicMock, patch
import sys

class TestAppUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock diffusers before app import
        cls.diffusers_patcher = patch.dict(sys.modules, {'diffusers': MagicMock()})
        cls.diffusers_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.diffusers_patcher.stop()

    def test_app_imports_and_has_css(self):
        import app
        self.assertIsNotNone(app.demo)
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

    def test_generate_image_validation(self):
        import app
        img, msg = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", msg)

if __name__ == '__main__':
    unittest.main()
