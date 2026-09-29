import unittest
from unittest.mock import MagicMock, patch

import torch


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline before importing app to avoid loading heavy weights
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe
        cls.mock_pipe.unet = MagicMock()
        cls.mock_pipe.vae = MagicMock()

        mock_image = MagicMock()
        cls.mock_result = MagicMock()
        cls.mock_result.images = [mock_image]
        cls.mock_pipe.return_value = cls.mock_result

        cls.pipeline_patcher = patch(
            "diffusers.StableDiffusionPipeline.from_pretrained",
            return_value=cls.mock_pipe,
        )
        cls.pipeline_patcher.start()

        import app
        cls.app = app

        # Save calls during import before any test setUp resets them
        cls.unet_to_calls = cls.mock_pipe.unet.to.call_args_list
        cls.vae_to_calls = cls.mock_pipe.vae.to.call_args_list

    @classmethod
    def tearDownClass(cls):
        cls.pipeline_patcher.stop()

    def setUp(self):
        self.mock_pipe.reset_mock()
        self.mock_pipe.to.return_value = self.mock_pipe
        self.mock_pipe.return_value = self.mock_result

    def test_app_imported_successfully(self):
        self.assertIsNotNone(self.app)

    def test_channels_last_applied(self):
        self.assertIn(
            unittest.mock.call(memory_format=torch.channels_last),
            self.unet_to_calls,
        )
        self.assertIn(
            unittest.mock.call(memory_format=torch.channels_last),
            self.vae_to_calls,
        )

    def test_generate_image_empty_prompt(self):
        image, status = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_success(self):
        # Track whether inference_mode is enabled during pipe execution
        inference_mode_active = []

        def side_effect(*args, **kwargs):
            inference_mode_active.append(torch.is_inference_mode_enabled())
            return self.mock_result

        self.mock_pipe.side_effect = side_effect

        image, status = self.app.generate_image(
            "a cute cat", "blurry", 20, 7.5, 512, 512, 42
        )

        self.assertIsNotNone(image)
        self.assertIn("Generated in", status)
        self.assertTrue(len(inference_mode_active) > 0)
        self.assertTrue(inference_mode_active[0])

    def test_generate_image_exception_handling(self):
        self.mock_pipe.side_effect = Exception("Model runtime error")

        image, status = self.app.generate_image(
            "a cat", "", 20, 7.5, 512, 512, -1
        )

        self.assertIsNone(image)
        self.assertIn("Error: Model runtime error", status)


if __name__ == "__main__":
    unittest.main()
