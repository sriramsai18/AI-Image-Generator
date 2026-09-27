import unittest
from unittest.mock import MagicMock, patch
from PIL import Image


class TestAppPerformanceAndFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up pipeline mock
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe
        cls.mock_pipe.unet = MagicMock()
        cls.mock_pipe.vae = MagicMock()

        # Fake output image
        fake_image = Image.new("RGB", (512, 512), color="blue")
        fake_result = MagicMock()
        fake_result.images = [fake_image]
        cls.mock_pipe.return_value = fake_result

        # Patch StableDiffusionPipeline before importing app
        cls.pipeline_patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=cls.mock_pipe)
        cls.mock_from_pretrained = cls.pipeline_patcher.start()

        # Import app
        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.pipeline_patcher.stop()

    def test_app_loaded_and_demo_exists(self):
        self.assertIsNotNone(self.app)
        self.assertIsNotNone(self.app.demo)

    def test_channels_last_applied(self):
        # Verify channels_last memory format was applied to unet and vae
        import torch
        self.mock_pipe.unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_pipe.vae.to.assert_called_with(memory_format=torch.channels_last)

    def test_generate_image_empty_prompt(self):
        image, status = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_valid_prompt(self):
        image, status = self.app.generate_image("a cyberpunk city", "blurry", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", status)
        self.assertIn("Steps: 25", status)
        self.assertIn("Seed: 42", status)
        self.mock_pipe.assert_called()

    def test_generate_image_exception_handling(self):
        self.mock_pipe.side_effect = Exception("CUDA error")
        image, status = self.app.generate_image("error prompt", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Error: CUDA error", status)
        self.mock_pipe.side_effect = None


if __name__ == "__main__":
    unittest.main()
