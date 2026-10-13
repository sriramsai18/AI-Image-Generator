import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestAppGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_pipe = MagicMock()
        cls.mock_image = Image.new("RGB", (512, 512))
        cls.mock_result = MagicMock()
        cls.mock_result.images = [cls.mock_image]
        cls.mock_pipe.return_value = cls.mock_result
        cls.mock_pipe.to.return_value = cls.mock_pipe

        cls.patcher = patch(
            "diffusers.StableDiffusionPipeline.from_pretrained",
            return_value=cls.mock_pipe,
        )
        cls.patcher.start()

        import app

        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_empty_prompt(self):
        img, status = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    def test_successful_generation(self):
        inference_mode_active = False

        def mock_call(*args, **kwargs):
            nonlocal inference_mode_active
            inference_mode_active = torch.is_inference_mode_enabled()
            return self.mock_result

        self.mock_pipe.side_effect = mock_call

        img, status = self.app.generate_image(
            "a beautiful sunset", "", 25, 7.5, 512, 512, 42
        )

        self.assertIsNotNone(img)
        self.assertTrue(inference_mode_active)
        self.assertIn("✅ Generated in", status)
        self.assertIn("Seed: 42", status)

    def test_generation_error_handling(self):
        self.mock_pipe.side_effect = Exception("GPU out of memory")

        img, status = self.app.generate_image(
            "a futuristic city", "", 25, 7.5, 512, 512, -1
        )

        self.assertIsNone(img)
        self.assertIn("❌ Error: GPU out of memory", status)


if __name__ == "__main__":
    unittest.main()
