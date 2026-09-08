import unittest
from unittest.mock import MagicMock, patch
import importlib
import sys

class TestApp(unittest.TestCase):
    def setUp(self):
        # Create a clean mock pipeline for diffusers
        self.mock_pipe = MagicMock()
        self.mock_pipe.to.return_value = self.mock_pipe

        # Patch diffusers before importing or reloading app
        self.patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=self.mock_pipe)
        self.mock_from_pretrained = self.patcher.start()

        # Import or reload app module
        if "app" in sys.modules:
            self.app = importlib.reload(sys.modules["app"])
        else:
            import app
            self.app = app

    def tearDown(self):
        self.patcher.stop()

    def test_generate_image_success(self):
        mock_image = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        self.mock_pipe.return_value = mock_result

        image, info = self.app.generate_image("a cute dog", "blurry", 25, 7.5, 512, 512, 42)

        self.assertEqual(image, mock_image)
        self.assertIn("✅ Generated in", info)
        self.assertIn("Steps: 25", info)

        # Verify pipeline was called with correct parameters
        self.mock_pipe.assert_called_once()
        kwargs = self.mock_pipe.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "a cute dog")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 25)
        self.assertEqual(kwargs["guidance_scale"], 7.5)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)

    def test_generate_image_empty_prompt(self):
        image, info = self.app.generate_image("   ", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertEqual(info, "⚠️ Please enter a prompt first!")

    def test_generate_image_exception(self):
        self.mock_pipe.side_effect = Exception("Out of memory")

        image, info = self.app.generate_image("a cat", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertEqual(info, "❌ Error: Out of memory")

if __name__ == "__main__":
    unittest.main()
