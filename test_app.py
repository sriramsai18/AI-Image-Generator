import unittest
from unittest.mock import MagicMock, patch
from PIL import Image


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create mock pipeline
        cls.mock_pipe = MagicMock()
        mock_result = MagicMock()
        dummy_img = Image.new("RGB", (512, 512), color="blue")
        mock_result.images = [dummy_img]
        cls.mock_pipe.return_value = mock_result
        cls.mock_pipe.to.return_value = cls.mock_pipe
        cls.mock_pipe.unet = MagicMock()

        # Patch StableDiffusionPipeline.from_pretrained
        cls.patcher = patch(
            "diffusers.StableDiffusionPipeline.from_pretrained",
            return_value=cls.mock_pipe,
        )
        cls.patcher.start()

        # Import app after patching
        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        self.mock_pipe.reset_mock()

    def test_app_imported(self):
        self.assertIsNotNone(self.app)

    def test_generate_image_empty_prompt(self):
        image, status = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt", status)

    def test_generate_image_success(self):
        mock_result = MagicMock()
        dummy_img = Image.new("RGB", (512, 512), color="red")
        mock_result.images = [dummy_img]

        in_inference_mode_during_call = []

        def mock_call(*args, **kwargs):
            import torch
            in_inference_mode_during_call.append(torch.is_inference_mode_enabled())
            return mock_result

        self.mock_pipe.side_effect = mock_call

        image, status = self.app.generate_image(
            prompt="a beautiful futuristic city",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertIsNotNone(image)
        self.assertIn("Generated in", status)
        self.assertTrue(len(in_inference_mode_during_call) > 0)
        self.assertTrue(in_inference_mode_during_call[0])


if __name__ == "__main__":
    unittest.main()
