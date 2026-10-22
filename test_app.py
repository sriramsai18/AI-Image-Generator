import importlib
import unittest
from unittest.mock import MagicMock, patch
import torch


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock pipeline object
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe
        cls.mock_unet = MagicMock()
        cls.mock_vae = MagicMock()
        cls.mock_pipe.unet = cls.mock_unet
        cls.mock_pipe.vae = cls.mock_vae

        # Mock result of pipeline call
        cls.mock_result = MagicMock()
        cls.mock_result.images = ["dummy_pil_image"]
        cls.mock_pipe.return_value = cls.mock_result

        # Patch StableDiffusionPipeline before importing app
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

    def setUp(self):
        self.mock_pipe.reset_mock()
        self.mock_pipe.to.return_value = self.mock_pipe
        self.mock_pipe.return_value = self.mock_result
        self.mock_pipe.side_effect = None

    def test_empty_prompt_returns_warning(self):
        image, info = self.app.generate_image(
            "", "negative prompt", 25, 7.5, 512, 512, -1
        )
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)

        image_whitespace, info_ws = self.app.generate_image(
            "   ", "negative prompt", 25, 7.5, 512, 512, -1
        )
        self.assertIsNone(image_whitespace)
        self.assertIn("Please enter a prompt first", info_ws)

    def test_successful_image_generation(self):
        image, info = self.app.generate_image(
            "a beautiful sunset",
            "blurry",
            25,
            7.5,
            512,
            512,
            42,
        )
        self.assertEqual(image, "dummy_pil_image")
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 25", info)
        self.assertIn("CFG: 7.5", info)
        self.assertIn("Seed: 42", info)

        self.mock_pipe.assert_called_once()
        _, kwargs = self.mock_pipe.call_args
        self.assertEqual(kwargs["prompt"], "a beautiful sunset")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 25)
        self.assertEqual(kwargs["guidance_scale"], 7.5)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)
        self.assertIsInstance(kwargs["generator"], torch.Generator)

    def test_image_generation_exception_handling(self):
        self.mock_pipe.side_effect = RuntimeError("CUDA out of memory")

        image, info = self.app.generate_image(
            "a cat", "", 20, 7.0, 512, 512, -1
        )
        self.assertIsNone(image)
        self.assertIn("Error: CUDA out of memory", info)

    def test_channels_last_applied_during_module_load(self):
        self.mock_unet.to.reset_mock()
        self.mock_vae.to.reset_mock()
        importlib.reload(self.app)
        self.mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_vae.to.assert_called_with(memory_format=torch.channels_last)

    def test_app_module_exists(self):
        self.assertIsNotNone(self.app)


if __name__ == "__main__":
    unittest.main()
