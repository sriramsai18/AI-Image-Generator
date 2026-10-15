import sys
import unittest
from unittest.mock import MagicMock

# Mock diffusers before importing app to avoid downloading model during tests
mock_diffusers = MagicMock()
mock_pipe = MagicMock()
mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipe
sys.modules["diffusers"] = mock_diffusers

import app


class TestAppFocusAccessibility(unittest.TestCase):
    def test_css_contains_focus_visible_styles(self):
        """Verify that app.css includes high-contrast focus-visible styles for accessibility."""
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important", app.css)
        self.assertIn("outline-offset: 2px !important", app.css)

    def test_demo_blocks_initialized(self):
        """Verify that the Gradio Blocks demo app initializes correctly."""
        self.assertIsNotNone(app.demo)


if __name__ == "__main__":
    unittest.main()
