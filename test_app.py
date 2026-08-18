import unittest
from unittest.mock import MagicMock, patch

class TestApp(unittest.TestCase):

    @patch("gradio.networking.url_ok", return_value=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_loading_and_generation(self, mock_cuda, mock_from_pretrained, mock_url_ok):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae
        mock_unet.to.return_value = mock_unet
        mock_vae.to.return_value = mock_vae
        mock_pipe.to.return_value = mock_pipe

        # Mock image output
        mock_result = MagicMock()
        mock_result.images = ["dummy_image"]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Import app or reload if already imported
        import app
        self.assertIsNotNone(app)

        # Test empty prompt
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

        # Test valid prompt generation
        img, info = app.generate_image("a photo of a cat", "blurry", 25, 7.5, 512, 512, 42)
        self.assertEqual(img, "dummy_image")
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 25", info)

        # Verify pipeline call parameters
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args[1]
        self.assertEqual(kwargs["prompt"], "a photo of a cat")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 25)
        self.assertEqual(kwargs["guidance_scale"], 7.5)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)
        self.assertIsNotNone(kwargs["generator"])

    @patch("gradio.networking.url_ok", return_value=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_error_handling(self, mock_cuda, mock_from_pretrained, mock_url_ok):
        mock_pipe = MagicMock()
        mock_pipe.side_effect = Exception("CUDA out of memory")
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        import app
        app.pipe = mock_pipe

        img, info = app.generate_image("a cat", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", info)

if __name__ == "__main__":
    unittest.main()
