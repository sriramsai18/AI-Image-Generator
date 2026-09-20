import sys
from unittest.mock import MagicMock
import unittest

# Mock diffusers and torch pipeline loading before app import
mock_pipeline_obj = MagicMock()
mock_pipeline_class = MagicMock()
mock_pipeline_class.from_pretrained.return_value = mock_pipeline_obj
mock_pipeline_obj.to.return_value = mock_pipeline_obj

sys.modules['diffusers'] = MagicMock()
sys.modules['diffusers'].StableDiffusionPipeline = mock_pipeline_class

import app  # noqa: E402

class TestApp(unittest.TestCase):

    def test_css_focus_visible_present(self):
        """Verify high-contrast focus visible rules are present in CSS."""
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

    def test_generate_image_empty_prompt(self):
        """Verify generate_image handles empty prompt gracefully."""
        img, msg = app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", msg)

    def test_generate_image_success(self):
        """Verify generate_image returns image when pipeline returns mock output."""
        mock_img = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_img]
        app.pipe.return_value = mock_result

        img, msg = app.generate_image("a cute cat", "", 20, 7.5, 512, 512, 42)
        self.assertEqual(img, mock_img)
        self.assertIn("Generated in", msg)
        self.assertIn("Seed: 42", msg)

if __name__ == "__main__":
    unittest.main()
