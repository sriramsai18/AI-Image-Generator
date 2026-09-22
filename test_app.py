import unittest
from unittest.mock import MagicMock, patch

import torch


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline.from_pretrained before app is imported
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe

        # Mock return value of pipeline call
        mock_output = MagicMock()
        mock_output.images = ["fake_pil_image"]
        cls.mock_pipe.return_value = mock_output

        cls.patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=cls.mock_pipe)
        cls.patcher.start()

        # Import app after patching
        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_generate_image_empty_prompt(self):
        image, info = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt", info)

    def test_generate_image_success(self):
        # Verify inference mode during execution
        is_inference_mode = []

        def side_effect(*args, **kwargs):
            is_inference_mode.append(torch.is_inference_mode_enabled())
            mock_output = MagicMock()
            mock_output.images = ["mock_image_result"]
            return mock_output

        self.mock_pipe.side_effect = side_effect

        image, info = self.app.generate_image("a beautiful sunset", "blurry", 25, 7.5, 512, 512, 42)

        self.assertEqual(image, "mock_image_result")
        self.assertIn("✅ Generated in", info)
        self.assertIn("Seed: 42", info)
        self.assertTrue(len(is_inference_mode) > 0 and is_inference_mode[0], "Inference mode should be enabled during pipeline call")

    def test_generate_image_exception(self):
        self.mock_pipe.side_effect = Exception("CUDA out of memory")
        image, info = self.app.generate_image("a beautiful sunset", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("❌ Error: CUDA out of memory", info)

if __name__ == "__main__":
    unittest.main()
