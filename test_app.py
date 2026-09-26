import unittest
from unittest.mock import MagicMock, patch

# Mock diffusers before importing app
with patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_from_pretrained:
    mock_pipe = MagicMock()
    mock_from_pretrained.return_value = mock_pipe
    from app import css, demo

class TestAppUX(unittest.TestCase):
    def test_css_contains_focus_visible(self):
        self.assertIn(":focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important;", css)
        self.assertIn(".gradio-container button:focus-visible", css)

    def test_demo_blocks_initialized(self):
        self.assertIsNotNone(demo)

if __name__ == "__main__":
    unittest.main()
