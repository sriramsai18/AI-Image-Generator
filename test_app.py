import unittest
from unittest.mock import MagicMock, patch
import sys

# Pre-mock StableDiffusionPipeline to prevent downloading weights during test import
mock_sd_pipeline = MagicMock()
mock_pipe_instance = MagicMock()
mock_sd_pipeline.from_pretrained.return_value = mock_pipe_instance
mock_pipe_instance.to.return_value = mock_pipe_instance

sys.modules['diffusers'] = MagicMock()
sys.modules['diffusers'].StableDiffusionPipeline = mock_sd_pipeline

import app  # noqa: E402

class TestApp(unittest.TestCase):

    def test_css_contains_focus_visible_styles(self):
        """Verify that high-contrast focus-visible styles are defined in CSS."""
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)

    def test_generate_image_empty_prompt(self):
        """Verify generate_image handles empty or whitespace prompt gracefully."""
        image, info = app.generate_image("", "negative", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first!", info)

    @patch("app.pipe")
    def test_generate_image_success(self, mock_pipe):
        """Verify generate_image generates image and formats status string correctly."""
        mock_output = MagicMock()
        mock_output.images = ["dummy_image"]
        mock_pipe.return_value = mock_output

        image, info = app.generate_image("a cyberpunk city", "blurry", 20, 7.0, 512, 512, 42)

        self.assertEqual(image, "dummy_image")
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 20", info)
        self.assertIn("CFG: 7.0", info)
        self.assertIn("Seed: 42", info)

if __name__ == "__main__":
    unittest.main()
