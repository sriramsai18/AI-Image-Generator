import unittest
from unittest.mock import MagicMock, patch
import torch

class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock pipeline before importing app
        cls.mock_unet = MagicMock()
        cls.mock_vae = MagicMock()
        cls.mock_pipe_obj = MagicMock()
        cls.mock_pipe_obj.to.return_value = cls.mock_pipe_obj
        cls.mock_pipe_obj.unet = cls.mock_unet
        cls.mock_pipe_obj.vae = cls.mock_vae

        cls.mock_result = MagicMock()
        cls.mock_result.images = ["fake_image_obj"]
        cls.mock_pipe_obj.return_value = cls.mock_result

        # Track if inference mode was enabled during call
        cls.inference_mode_enabled_during_call = False

        def pipe_side_effect(*args, **kwargs):
            cls.inference_mode_enabled_during_call = torch.is_inference_mode_enabled()
            return cls.mock_result

        cls.mock_pipe_obj.side_effect = pipe_side_effect

        cls.patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=cls.mock_pipe_obj)
        cls.patcher.start()

        # Import app after patching from_pretrained
        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_app_imported(self):
        self.assertIsNotNone(self.app)

    def test_channels_last_applied(self):
        self.mock_unet.to.assert_called_with(memory_format=torch.channels_last)
        self.mock_vae.to.assert_called_with(memory_format=torch.channels_last)

    def test_generate_image_empty_prompt(self):
        image, status = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_valid_prompt_uses_inference_mode(self):
        TestApp.inference_mode_enabled_during_call = False
        image, status = self.app.generate_image("a cute cat", "blurry", 25, 7.5, 512, 512, 42)

        self.assertIsNotNone(image)
        self.assertIn("Generated in", status)
        self.assertTrue(TestApp.inference_mode_enabled_during_call, "torch.inference_mode() should be enabled during pipeline execution")

    def test_generate_image_error_handling(self):
        # Force pipe call to raise exception
        self.mock_pipe_obj.side_effect = Exception("CUDA Out of Memory")
        image, status = self.app.generate_image("error test", "", 25, 7.5, 512, 512, -1)

        self.assertIsNone(image)
        self.assertIn("Error: CUDA Out of Memory", status)

        # Restore side effect
        def pipe_side_effect(*args, **kwargs):
            TestApp.inference_mode_enabled_during_call = torch.is_inference_mode_enabled()
            return self.mock_result
        self.mock_pipe_obj.side_effect = pipe_side_effect

if __name__ == "__main__":
    unittest.main()
