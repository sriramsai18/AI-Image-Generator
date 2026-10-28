import os
import unittest
from unittest.mock import patch, MagicMock

# Mock os.sysconf for cross-platform compatibility if needed
orig_sysconf = getattr(os, 'sysconf', None)


def mock_sysconf(name):
    if name == 'SC_PAGE_SIZE':
        return 4096
    if name == 'SC_PHYS_PAGES':
        return 2097152
    if orig_sysconf is not None:
        return orig_sysconf(name)
    raise ValueError(f"Unknown sysconf key: {name}")


os.sysconf = mock_sysconf

# Mock Gradio's url_ok to avoid localhost networking issues
import gradio.networking  # noqa: E402

gradio.networking.url_ok = lambda *args, **kwargs: True


class TestAppAccessibilityAndLogic(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_css_and_initialization(self, mock_cuda, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        self.assertIsNotNone(app.demo)

        # Verify custom CSS includes focus-visible rules
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn("outline: 2px solid #39ff14 !important;", app.css)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_empty_prompt(self, mock_cuda, mock_from_pretrained):
        import app
        img, status = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)


if __name__ == "__main__":
    unittest.main()
