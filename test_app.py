import unittest
from unittest.mock import MagicMock, patch
import PIL.Image

# Mock heavy model loading before app is imported
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe

patcher = patch('diffusers.StableDiffusionPipeline.from_pretrained', return_value=mock_pipe)
patcher.start()

import app

class TestAppGeneration(unittest.TestCase):
    def tearDown(self):
        mock_pipe.reset_mock()

    @patch('app.torch.inference_mode')
    def test_generate_image_success(self, mock_inference_mode):
        # Setup mock return image
        mock_image = PIL.Image.new("RGB", (512, 512), color="red")
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        app.pipe.return_value = mock_result

        # Call generate_image
        image, info = app.generate_image(
            prompt="a glowing futuristic city",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        # Assertions
        self.assertEqual(image, mock_image)
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 20", info)
        app.pipe.assert_called_once()
        mock_inference_mode.assert_called_once()

    def test_generate_image_empty_prompt(self):
        app.pipe.reset_mock()
        image, info = app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )

        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)
        app.pipe.assert_not_called()

if __name__ == "__main__":
    unittest.main()
