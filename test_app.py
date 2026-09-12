import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


class TestAppGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline.from_pretrained before importing app
        cls.patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained")
        cls.mock_from_pretrained = cls.patcher.start()

        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe
        mock_image = MagicMock()
        cls.mock_pipe.return_value.images = [mock_image]
        cls.mock_from_pretrained.return_value = cls.mock_pipe

        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_generate_image_valid_prompt(self):
        img, info = self.app.generate_image("a futuristic city", "blurry", 20, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 20", info)
        self.assertIn("Seed: 42", info)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", info)

    def test_generate_image_exception_handling(self):
        self.app.pipe.side_effect = RuntimeError("GPU OOM")
        img, info = self.app.generate_image("a prompt", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: GPU OOM", info)
        self.app.pipe.side_effect = None


if __name__ == "__main__":
    unittest.main()
