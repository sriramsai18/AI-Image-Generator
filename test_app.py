import os
import unittest
from unittest.mock import patch

os.environ["MOCK_MODE"] = "1"
import app


class TestAppUXAndAccessibility(unittest.TestCase):
    def test_focus_visible_css_present(self):
        """Verify that high-contrast focus-visible styles exist for keyboard accessibility."""
        css = app.css
        self.assertIn(":focus-visible", css)
        self.assertIn(".gradio-container button:focus-visible", css)
        self.assertIn(".gradio-container input:focus-visible", css)
        self.assertIn(".gradio-container textarea:focus-visible", css)
        self.assertIn(".gradio-container a:focus-visible", css)
        self.assertIn(".gradio-container [tabindex]:focus-visible", css)
        self.assertIn("outline: 2px solid #00f0ff !important", css)
        self.assertIn("box-shadow: 0 0 10px rgba(0, 240, 255, 0.6) !important", css)

    def test_footer_links_rel_noopener(self):
        """Verify external footer links include security and accessibility attributes."""
        found_rel = False
        for block in app.demo.blocks.values():
            if hasattr(block, "value") and isinstance(block.value, str):
                if "app-footer" in block.value:
                    self.assertIn('rel="noopener noreferrer"', block.value)
                    found_rel = True
        self.assertTrue(found_rel, "Footer HTML block with rel='noopener noreferrer' not found")

    def test_mock_pipeline_returns_image(self):
        """Verify MockPipeline generates an image with correct dimensions."""
        pipe = app.MockPipeline()
        result = pipe("a futuristic cyberpunk city", width=256, height=256)
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.images[0].size, (256, 256))

    def test_generate_image_empty_prompt(self):
        """Verify generate_image handles empty or whitespace-only prompts gracefully."""
        image, status = app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_valid_prompt(self):
        """Verify generate_image returns generated image and metadata status."""
        image, status = app.generate_image(
            "a futuristic cyberpunk city at night", "blurry", 25, 7.5, 512, 512, 42
        )
        self.assertIsNotNone(image)
        self.assertIn("Generated in", status)
        self.assertIn("Steps: 25", status)
        self.assertIn("CFG: 7.5", status)
        self.assertIn("Seed: 42", status)

    def test_generate_image_random_seed(self):
        """Verify random seed (-1) generates properly formatted status."""
        image, status = app.generate_image(
            "a lone tree on a hill", "blurry", 20, 7.0, 512, 512, -1
        )
        self.assertIsNotNone(image)
        self.assertIn("Seed: random", status)

    def test_pipe_exception_handling(self):
        """Verify pipeline exceptions are handled cleanly without crashing."""
        with patch.object(app, "pipe", side_effect=RuntimeError("GPU OOM")):
            image, status = app.generate_image("crash test", "", 20, 7.0, 512, 512, -1)
            self.assertIsNone(image)
            self.assertIn("❌ Error: GPU OOM", status)

    @patch("gradio.networking.url_ok", return_value=True)
    def test_demo_blocks_instance(self, mock_url_ok):
        """Verify Gradio demo instance is initialized with correct title."""
        self.assertIsNotNone(app.demo)
        self.assertIn("Text2Image", app.demo.title)


if __name__ == "__main__":
    unittest.main()
