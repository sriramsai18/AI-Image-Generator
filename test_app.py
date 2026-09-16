import unittest
from unittest.mock import MagicMock, patch


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create mock pipeline
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe

        # Mock result for pipeline call
        mock_result = MagicMock()
        mock_result.images = ["fake_image_object"]
        cls.mock_pipe.return_value = mock_result

        # Patch StableDiffusionPipeline before importing app
        cls.patcher = patch(
            "diffusers.StableDiffusionPipeline.from_pretrained",
            return_value=cls.mock_pipe,
        )
        cls.mock_from_pretrained = cls.patcher.start()

        # Import app module
        import app

        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_app_loaded(self):
        self.assertIsNotNone(self.app)

    def test_channels_last_applied(self):
        self.mock_pipe.unet.to.assert_called_with(memory_format=self.app.torch.channels_last)
        self.mock_pipe.vae.to.assert_called_with(memory_format=self.app.torch.channels_last)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image("", "neg", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt", info)

    def test_generate_image_valid(self):
        img, info = self.app.generate_image("a dog", "", 20, 7.5, 512, 512, 42)
        self.assertEqual(img, "fake_image_object")
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)


if __name__ == "__main__":
    unittest.main()
