import sys
import unittest
from unittest.mock import MagicMock

# Mock modules before loading app
mock_pipe = MagicMock()
mock_diffusers = MagicMock()
mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipe

sys.modules['diffusers'] = mock_diffusers

import gradio as gr  # noqa: E402
from app import css, demo  # noqa: E402

class TestAppAccessibilityCSS(unittest.TestCase):
    def test_css_contains_focus_visible_rules(self):
        self.assertIn(".gradio-container button:focus-visible", css)
        self.assertIn(".gradio-container input:focus-visible", css)
        self.assertIn(".gradio-container textarea:focus-visible", css)
        self.assertIn(".gradio-container a:focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important;", css)
        self.assertIn("box-shadow: 0 0 0 4px rgba(230, 57, 70, 0.4) !important;", css)

    def test_demo_blocks_initialized(self):
        self.assertIsInstance(demo, gr.Blocks)

if __name__ == "__main__":
    unittest.main()
