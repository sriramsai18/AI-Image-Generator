import unittest
from unittest.mock import MagicMock, patch


class TestApp(unittest.TestCase):
    @patch("app.pipe")
    def test_generate_image_empty_prompt(self, mock_pipe):
        from app import generate_image

        image, info = generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)
        mock_pipe.assert_not_called()

    @patch("app.pipe")
    def test_generate_image_success(self, mock_pipe):
        from app import generate_image

        mock_result = MagicMock()
        mock_result.images = ["dummy_image"]
        mock_pipe.return_value = mock_result

        image, info = generate_image("a beautiful cat", "ugly", 25, 7.5, 512, 512, 42)

        self.assertEqual(image, "dummy_image")
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 25", info)
        self.assertIn("Seed: 42", info)
        mock_pipe.assert_called_once()

    @patch("app.pipe")
    def test_generate_image_error_handling(self, mock_pipe):
        from app import generate_image

        mock_pipe.side_effect = RuntimeError("GPU out of memory")

        image, info = generate_image("test prompt", "", 25, 7.5, 512, 512, -1)

        self.assertIsNone(image)
        self.assertIn("Error: GPU out of memory", info)


if __name__ == "__main__":
    unittest.main()
