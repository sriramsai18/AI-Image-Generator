import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

# Setup mock pipeline prior to importing app
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe
mock_pipe.enable_attention_slicing.return_value = None

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe), \
     patch("torch.cuda.is_available", return_value=False), \
     patch("gradio.networking.url_ok", return_value=True):
    import app


class TestApp(unittest.TestCase):
    def setUp(self):
        mock_pipe.reset_mock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.enable_attention_slicing.return_value = None

    def test_app_loads(self):
        """Ensure demo Blocks app is initialized properly."""
        self.assertIsNotNone(app.demo)
        self.assertEqual(app.demo.title, "Text2Image — Sriram")

    def test_empty_prompt_validation(self):
        """Empty prompts should return validation warning without invoking pipeline."""
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", info)
        mock_pipe.assert_not_called()

    def test_successful_generation(self):
        """Valid prompt generates an image and returns formatted status."""
        test_img = Image.new("RGB", (512, 512), color="purple")
        mock_res = MagicMock()
        mock_res.images = [test_img]
        mock_pipe.return_value = mock_res

        img, info = app.generate_image("a cute cat in neon city", "", 20, 7.0, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("✅ Generated in", info)
        self.assertIn("Seed: 42", info)
        mock_pipe.assert_called_once()

    def test_generation_error_handling(self):
        """Exceptions during generation are gracefully handled and displayed in status."""
        mock_pipe.side_effect = RuntimeError("CUDA out of memory")

        img, info = app.generate_image("test error", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("❌ Error: CUDA out of memory", info)
        mock_pipe.side_effect = None

    def test_css_focus_visible_and_button_styles(self):
        """Check that CSS includes keyboard accessibility focus-visible rules and secondary button styling."""
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn(".gradio-container input:focus-visible", app.css)
        self.assertIn(".gradio-container textarea:focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", app.css)
        self.assertIn("button.secondary-btn", app.css)


if __name__ == "__main__":
    unittest.main()
