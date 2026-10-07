import unittest
from unittest.mock import MagicMock, patch

class TestAppPerformance(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_inference_mode(self, mock_cuda, mock_from_pretrained):
        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe

        # Mock result of pipeline call
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        import app

        # Test empty prompt error handling
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt", info)

        # Test valid generation
        img, info = app.generate_image("a cyberpunk city", "blurry", 25, 7.5, 512, 512, 42)
        self.assertEqual(img, mock_image)
        self.assertIn("Generated in", info)

        # Verify pipeline was called
        mock_pipe.assert_called_once()
        _, kwargs = mock_pipe.call_args
        self.assertEqual(kwargs["prompt"], "a cyberpunk city")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 25)

if __name__ == "__main__":
    unittest.main()
