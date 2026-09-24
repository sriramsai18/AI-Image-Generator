import unittest
from unittest.mock import MagicMock, patch

class TestApp(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_validation_and_success(self, mock_cuda, mock_from_pretrained):
        # Mock the pipeline instance
        mock_pipe_instance = MagicMock()
        mock_from_pretrained.return_value = mock_pipe_instance
        mock_pipe_instance.to.return_value = mock_pipe_instance

        # Mock generated output
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe_instance.return_value = mock_result

        # Import app after patching
        import app

        # Test empty prompt validation
        img, status = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", status)

        # Test valid generation call
        img, status = app.generate_image("a cyberpunk cat", "blurry", 25, 7.5, 512, 512, 42)
        self.assertEqual(img, mock_image)
        self.assertIn("Generated in", status)
        self.assertIn("Seed: 42", status)

if __name__ == "__main__":
    unittest.main()
