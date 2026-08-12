# ruff: noqa: E402
import sys
import unittest
from unittest.mock import MagicMock

# Mock out heavy imports/operations during module loading
mock_pipeline_class = MagicMock()
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe
mock_pipeline_class.from_pretrained.return_value = mock_pipe

# Patch StableDiffusionPipeline before importing app.py
sys.modules['diffusers'] = MagicMock()
import diffusers
diffusers.StableDiffusionPipeline = mock_pipeline_class

# Now we can import app
import app

class TestApp(unittest.TestCase):

    def setUp(self):
        # Reset mocks and side_effects/return_values
        mock_pipeline_class.reset_mock()
        mock_pipe.reset_mock()
        mock_pipe.side_effect = None
        mock_pipe.to.return_value = mock_pipe

    def test_custom_css_focus_visible_included(self):
        """Verify that high-contrast accessibility focus styles are defined in CSS."""
        self.assertIn(".gradio-container textarea:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important", app.css)

    def test_generate_image_empty_prompt(self):
        """Verify that generate_image handles empty prompts gracefully."""
        image, status = app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("⚠️ Please enter a prompt first!", status)

    def test_generate_image_success(self):
        """Verify that generate_image correctly calls pipe and returns an image."""
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        image, status = app.generate_image(
            prompt="a beautiful cat",
            negative_prompt="blurry",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        self.assertEqual(image, mock_image)
        self.assertIn("✅ Generated in", status)
        self.assertIn("Seed: 42", status)

    def test_generate_image_exception(self):
        """Verify that exceptions in generation function are caught and status shows the error."""
        mock_pipe.side_effect = Exception("CUDA out of memory")

        image, status = app.generate_image(
            prompt="a beautiful cat",
            negative_prompt="blurry",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )

        self.assertIsNone(image)
        self.assertIn("❌ Error: CUDA out of memory", status)

if __name__ == "__main__":
    unittest.main()
