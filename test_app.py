import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock pipeline instance
        cls.mock_pipe_instance = MagicMock()
        cls.mock_pipe_instance.to.return_value = cls.mock_pipe_instance
        cls.mock_unet = MagicMock()
        cls.mock_vae = MagicMock()
        cls.mock_pipe_instance.unet = cls.mock_unet
        cls.mock_pipe_instance.vae = cls.mock_vae

        # Mock image return
        mock_image = Image.new("RGB", (512, 512), color="blue")
        cls.mock_result = MagicMock()
        cls.mock_result.images = [mock_image]
        cls.mock_pipe_instance.return_value = cls.mock_result

        # Patch pipeline load before importing app
        cls.patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=cls.mock_pipe_instance)
        cls.patcher.start()

        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_empty_prompt_returns_warning(self):
        img, status = self.app.generate_image("", "ugly", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", status)

    def test_whitespace_prompt_returns_warning(self):
        img, status = self.app.generate_image("   ", "ugly", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", status)

    def test_valid_prompt_generation(self):
        img, status = self.app.generate_image("a cute cat", "ugly", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)
        self.assertIn("Steps: 25", status)

if __name__ == "__main__":
    unittest.main()
