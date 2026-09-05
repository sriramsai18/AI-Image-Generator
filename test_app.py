import unittest
import sys
from unittest.mock import MagicMock

# Mock required dependencies before importing app
mock_gradio = MagicMock()
sys.modules['gradio'] = mock_gradio

mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
sys.modules['torch'] = mock_torch

mock_diffusers = MagicMock()
mock_pipe = MagicMock()
mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipe
mock_pipe.to.return_value = mock_pipe
sys.modules['diffusers'] = mock_diffusers

import app  # noqa: E402

class TestAppAccessibility(unittest.TestCase):
    def test_focus_visible_css_present(self):
        """Test that custom CSS includes focus-visible accessibility rules."""
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)
        self.assertIn(".gradio-container textarea:focus-visible", app.css)
        self.assertIn(".gradio-container a:focus-visible", app.css)
        self.assertIn("outline: 2px solid #39ff14 !important;", app.css)

    def test_empty_prompt_validation(self):
        """Test that generate_image validates empty prompt input."""
        img, status = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", status)

if __name__ == '__main__':
    unittest.main()
