import unittest
from unittest.mock import MagicMock, patch

# Mock diffusers before importing app module to avoid downloading model weights
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe
mock_result = MagicMock()
mock_result.images = ["mock_pil_image"]
mock_pipe.return_value = mock_result

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
    import app

class TestApp(unittest.TestCase):

    def setUp(self):
        # Reset mock pipe return value and side_effect before each test
        app.pipe.side_effect = None
        app.pipe.return_value = mock_result

    def test_generate_image_empty_prompt(self):
        img, status = app.generate_image("   ", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_success(self):
        img, status = app.generate_image("a cyberpunk city", "blurry", 25, 7.5, 512, 512, 42)
        self.assertEqual(img, "mock_pil_image")
        self.assertIn("Generated in", status)
        self.assertIn("Seed: 42", status)

    def test_generate_image_random_seed(self):
        img, status = app.generate_image("a cozy cafe", "", 20, 7.0, 256, 256, -1)
        self.assertEqual(img, "mock_pil_image")
        self.assertIn("Seed: random", status)

    def test_generate_image_exception_handling(self):
        app.pipe.side_effect = RuntimeError("Out of memory")
        img, status = app.generate_image("a futuristic car", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: Out of memory", status)

if __name__ == "__main__":
    unittest.main()
