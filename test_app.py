import unittest
from unittest.mock import MagicMock, patch
import app

class TestApp(unittest.TestCase):
    def test_generate_image_empty_prompt(self):
        img, msg = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", msg)

    @patch("app.pipe")
    def test_generate_image_valid_prompt(self, mock_pipe):
        mock_output = MagicMock()
        mock_output.images = ["fake_image"]
        mock_pipe.return_value = mock_output

        img, msg = app.generate_image("a cute cat", "blurry", 20, 7.5, 512, 512, 42)
        self.assertEqual(img, "fake_image")
        self.assertIn("Generated in", msg)
        self.assertIn("Seed: 42", msg)

if __name__ == "__main__":
    unittest.main()
