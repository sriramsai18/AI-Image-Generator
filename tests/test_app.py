import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock StableDiffusionPipeline before importing app to avoid loading the real model during tests
mock_pipe_instance = MagicMock()
mock_from_pretrained = MagicMock(return_value=mock_pipe_instance)

# .to() should return the pipe itself so mock references remain consistent
mock_pipe_instance.to.return_value = mock_pipe_instance

with patch("diffusers.StableDiffusionPipeline.from_pretrained", mock_from_pretrained):
    import app


class TestAppCorrectness(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        mock_pipe_instance.reset_mock(side_effect=True)
        mock_from_pretrained.reset_mock()
        # Ensure to() still returns the pipe itself after resetting
        mock_pipe_instance.to.return_value = mock_pipe_instance

    def test_empty_prompt(self):
        """Test that generation fails immediately when prompt is empty or just spaces."""
        image, status = app.generate_image("", "ugly", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

        image2, status2 = app.generate_image("   ", "ugly", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image2)
        self.assertIn("Please enter a prompt first", status2)

    def test_generate_image_success(self):
        """Test that generate_image correctly invokes the pipe with arguments and returns simulated image."""
        # Setup simulated result from pipeline
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe_instance.return_value = mock_result

        # Call generate_image
        image, status = app.generate_image(
            "cyberpunk city", "blurry", 25, 7.5, 512, 512, 1234
        )

        # Assertions
        self.assertEqual(image, mock_image)
        self.assertIn("Generated in", status)
        self.assertIn("Steps: 25", status)
        self.assertIn("CFG: 7.5", status)
        self.assertIn("Seed: 1234", status)

        # Verify how the mock pipeline was called
        mock_pipe_instance.assert_called_once()
        kwargs = mock_pipe_instance.call_args[1]
        self.assertEqual(kwargs["prompt"], "cyberpunk city")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 25)
        self.assertEqual(kwargs["guidance_scale"], 7.5)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)
        self.assertIsNotNone(kwargs["generator"])

    def test_generate_image_handles_exception(self):
        """Test that exceptions raised by the pipeline are gracefully handled and returned in status."""
        mock_pipe_instance.side_effect = RuntimeError(
            "Something went wrong during generation"
        )

        image, status = app.generate_image("cat", "blurry", 10, 5.0, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Error: Something went wrong during generation", status)


if __name__ == "__main__":
    unittest.main()
