import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# Mock StableDiffusionPipeline before importing app
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe
mock_result = MagicMock()
dummy_img = Image.new("RGB", (512, 512), color="red")
mock_result.images = [dummy_img]
mock_pipe.return_value = mock_result

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
    import app

class TestApp(unittest.TestCase):
    def test_app_loaded(self):
        self.assertIsNotNone(app.pipe)

    def test_generate_image_empty_prompt(self):
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    def test_generate_image_success(self):
        img, info = app.generate_image(
            "a futuristic cityscape",
            "blurry",
            25,
            7.5,
            512,
            512,
            42
        )
        self.assertEqual(img, dummy_img)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

    def test_generate_image_random_seed(self):
        img, info = app.generate_image(
            "a lone tree",
            "",
            20,
            7.0,
            512,
            512,
            -1
        )
        self.assertEqual(img, dummy_img)
        self.assertIn("Seed: random", info)

if __name__ == "__main__":
    unittest.main()
