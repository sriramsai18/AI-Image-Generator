import unittest
from unittest.mock import MagicMock, patch

# Mock StableDiffusionPipeline.from_pretrained before importing app
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe

# Mock pipeline call returning a result with a dummy image
mock_result = MagicMock()
mock_result.images = ["dummy_image"]
mock_pipe.return_value = mock_result

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipe):
    import app


class TestApp(unittest.TestCase):
    def test_app_imported(self):
        self.assertIsNotNone(app)
        self.assertIsNotNone(app.demo)

    def test_generate_image_empty_prompt(self):
        image, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt", info)

    def test_generate_image_valid_prompt(self):
        image, info = app.generate_image(
            "a beautiful sunset",
            "blurry",
            25,
            7.5,
            512,
            512,
            42,
        )
        self.assertEqual(image, "dummy_image")
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

    def test_generate_image_random_seed(self):
        image, info = app.generate_image(
            "cyberpunk city",
            "low quality",
            30,
            8.0,
            512,
            512,
            -1,
        )
        self.assertEqual(image, "dummy_image")
        self.assertIn("Seed: random", info)


if __name__ == "__main__":
    unittest.main()
