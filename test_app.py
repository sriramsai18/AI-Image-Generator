import unittest
from unittest.mock import MagicMock, patch
import app


class TestGenerateImage(unittest.TestCase):
    def test_empty_prompt_returns_warning(self):
        img, status = app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    @patch("app.pipe")
    def test_successful_generation(self, mock_pipe):
        mock_result = MagicMock()
        mock_result.images = ["fake_pil_image"]
        mock_pipe.return_value = mock_result

        img, status = app.generate_image(
            prompt="a serene lake",
            negative_prompt="blurry",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertEqual(img, "fake_pil_image")
        self.assertIn("Generated in", status)
        self.assertIn("Seed: 42", status)
        mock_pipe.assert_called_once()

    @patch("app.pipe")
    def test_error_handling(self, mock_pipe):
        mock_pipe.side_effect = Exception("CUDA out of memory")

        img, status = app.generate_image(
            prompt="a detailed prompt",
            negative_prompt="",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )

        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", status)


if __name__ == "__main__":
    unittest.main()
