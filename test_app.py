import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_empty_prompt(self, mock_cuda, mock_from_pretrained):
        import app
        self.assertIsNotNone(app)

        image, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first!", info)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_success(self, mock_cuda, mock_from_pretrained):
        # Create a mock pipeline return value
        dummy_image = Image.new("RGB", (512, 512))
        mock_result = MagicMock()
        mock_result.images = [dummy_image]

        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_result
        mock_pipe.to.return_value = mock_pipe
        app_pipe_patch = patch("app.pipe", mock_pipe)

        with app_pipe_patch:
            import app
            image, info = app.generate_image("a beautiful sunset", "blurry", 25, 7.5, 512, 512, 42)
            self.assertEqual(image, dummy_image)
            self.assertIn("✅ Generated in", info)
            self.assertIn("Seed: 42", info)
            mock_pipe.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_exception_handling(self, mock_cuda, mock_from_pretrained):
        mock_pipe = MagicMock()
        mock_pipe.side_effect = RuntimeError("GPU out of memory")
        mock_pipe.to.return_value = mock_pipe

        with patch("app.pipe", mock_pipe):
            import app
            image, info = app.generate_image("a cat", "", 25, 7.5, 512, 512, -1)
            self.assertIsNone(image)
            self.assertIn("❌ Error: GPU out of memory", info)

if __name__ == "__main__":
    unittest.main()
