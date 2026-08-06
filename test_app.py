import unittest
from unittest.mock import patch, MagicMock

# Mock out dependencies to prevent actual model downloading
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe

@patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe)
@patch("gradio.networking.url_ok", return_value=True)
class TestApp(unittest.TestCase):

    def test_app_layout(self, mock_url_ok, mock_from_pretrained):
        # Import app to trigger model loading and layout building
        import app
        self.assertIsNotNone(app.demo)
        # Check custom css holds focus-visible and secondary button styles
        self.assertIn("focus-visible", app.css)
        self.assertIn("button.secondary", app.css)

    def test_reset_all_fields(self, mock_url_ok, mock_from_pretrained):
        import app
        reset_vals = app.reset_all_fields()
        # Returns tuple of reset values:
        # prompt, negative_prompt, steps, guidance, width, height, seed, output_image, info_text
        self.assertEqual(reset_vals[0], "")
        self.assertEqual(reset_vals[1], "blurry, ugly, distorted, low quality, watermark")
        self.assertEqual(reset_vals[2], 25)
        self.assertEqual(reset_vals[3], 7.5)
        self.assertEqual(reset_vals[4], 512)
        self.assertEqual(reset_vals[5], 512)
        self.assertEqual(reset_vals[6], -1)
        self.assertIsNone(reset_vals[7])
        self.assertEqual(reset_vals[8], "")

if __name__ == "__main__":
    unittest.main()
