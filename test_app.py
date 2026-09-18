import unittest
from unittest.mock import MagicMock, patch
import torch

class TestApp(unittest.TestCase):

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def setUp(self, mock_from_pretrained):
        self.mock_pipe = MagicMock()
        self.mock_pipe.device = torch.device("cpu")
        self.mock_pipe.to.return_value = self.mock_pipe
        mock_from_pretrained.return_value = self.mock_pipe

        # Import app after patching
        import app
        self.app = app
        self.app.pipe = self.mock_pipe

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    def test_generate_image_success(self):
        mock_result = MagicMock()
        mock_result.images = ["fake_image"]
        self.mock_pipe.return_value = mock_result

        img, info = self.app.generate_image("a cute cat", "blurry", 20, 7.0, 512, 512, 42)

        self.assertEqual(img, "fake_image")
        self.assertIn("✅ Generated in", info)
        self.assertIn("Seed: 42", info)
        self.mock_pipe.assert_called_once()

    def test_generate_image_exception(self):
        self.mock_pipe.side_effect = RuntimeError("GPU out of memory")

        img, info = self.app.generate_image("a cute cat", "", 20, 7.0, 512, 512, -1)

        self.assertIsNone(img)
        self.assertIn("❌ Error: GPU out of memory", info)

if __name__ == "__main__":
    unittest.main()
