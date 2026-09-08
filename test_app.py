import unittest
from unittest.mock import MagicMock, patch

class TestAppUX(unittest.TestCase):
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('torch.cuda.is_available', return_value=False)
    def test_app_css_contains_focus_visible(self, mock_cuda, mock_sd):
        mock_pipe = MagicMock()
        mock_sd.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        import app
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn("outline: 2px solid #39ff14 !important;", app.css)

if __name__ == '__main__':
    unittest.main()
