import sys
from unittest.mock import MagicMock

# Mock stable diffusion loading to run unit tests in any environment quickly and without GPU / internet
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe

# Let's apply a patch at module import level so we NEVER trigger runwayml/stable-diffusion-v1-5 downloading when importing app
sys.modules["diffusers"] = MagicMock()
sys.modules["diffusers"].StableDiffusionPipeline = MagicMock()
sys.modules["diffusers"].StableDiffusionPipeline.from_pretrained.return_value = mock_pipe

import app  # noqa: E402
import unittest  # noqa: E402

class TestImageGeneratorApp(unittest.TestCase):

    def test_generate_image_empty_prompt(self):
        # Prompt cannot be empty
        image, status = app.generate_image("", "ugly", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("⚠️ Please enter a prompt first!", status)

    def test_generate_image_with_custom_seed(self):
        # Positive generation flow with specified seed
        dummy_image = MagicMock()
        mock_pipe.return_value.images = [dummy_image]

        image, status = app.generate_image("a futuristic neon city", "blurry", 20, 8.0, 512, 512, 12345)
        self.assertEqual(image, dummy_image)
        self.assertIn("✅ Generated in", status)
        self.assertIn("Seed: 12345", status)

    def test_generate_image_exception_handling(self):
        # Check if exceptions inside pipe call are caught and returned properly
        mock_pipe.side_effect = RuntimeError("Out of VRAM")

        image, status = app.generate_image("impossible design", "", 25, 7.0, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("❌ Error: Out of VRAM", status)
        # reset side effect for other tests
        mock_pipe.side_effect = None

    def test_custom_css_focus_visible_included(self):
        # Custom CSS should contain our custom focus-visible visual rules
        self.assertIn("*:focus-visible", app.css)
        self.assertIn("outline: 3px solid #ffaa00", app.css)

if __name__ == "__main__":
    unittest.main()
