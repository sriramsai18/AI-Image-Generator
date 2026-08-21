import importlib
import sys
import unittest
from unittest.mock import MagicMock

# Mock torch, diffusers, gradio before importing app
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
sys.modules["torch"] = mock_torch

mock_diffusers = MagicMock()
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe
mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipe
sys.modules["diffusers"] = mock_diffusers

mock_gradio = MagicMock()
sys.modules["gradio"] = mock_gradio

app = importlib.import_module("app")


class TestAppAccessibilityAndUI(unittest.TestCase):

    def test_css_focus_visible_styling(self):
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)
        self.assertIn("outline-offset: 2px !important;", app.css)
        self.assertIn(
            "box-shadow: 0 0 12px rgba(230, 57, 70, 0.6) !important;", app.css
        )

    def test_generate_image_validation(self):
        img, msg = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", msg)


if __name__ == "__main__":
    unittest.main()
