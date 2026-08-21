import sys
import unittest
from unittest.mock import MagicMock

import PIL.Image

# Pre-mock diffusers.StableDiffusionPipeline before importing app
mock_pipe_base = MagicMock()
mock_pipe_loaded = MagicMock()
mock_pipe_base.to.return_value = mock_pipe_loaded

mock_from_pretrained = MagicMock(return_value=mock_pipe_base)

sys.modules["diffusers"] = MagicMock()
sys.modules["diffusers"].StableDiffusionPipeline.from_pretrained = mock_from_pretrained


class TestAppPerformanceOptimizations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Import app after mocking diffusers
        import app
        cls.app = app

    def setUp(self):
        # Reset mock_pipe_loaded side effect and return value before each test
        mock_result = MagicMock()
        mock_result.images = [PIL.Image.new("RGB", (64, 64), color="red")]
        mock_pipe_loaded.side_effect = None
        mock_pipe_loaded.return_value = mock_result

    def test_empty_prompt(self):
        image, status = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

    def test_valid_prompt_generation(self):
        prompt = "a cute robot running fast"
        image, status = self.app.generate_image(prompt, "blurry", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("✅ Generated in", status)
        self.assertIn("Seed: 42", status)

    def test_generation_exception_handling(self):
        mock_pipe_loaded.side_effect = RuntimeError("CUDA out of memory")
        image, status = self.app.generate_image("a test prompt", "", 20, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("❌ Error: CUDA out of memory", status)

    def test_demo_blocks_initialized(self):
        self.assertIsNotNone(self.app.demo)


if __name__ == "__main__":
    unittest.main()
