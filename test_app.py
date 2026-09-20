import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock pipeline before importing app to avoid downloading heavy model weights during tests
        cls.mock_pipe = MagicMock()
        # Mock chaining methods like .to('cpu') or .to('cuda')
        cls.mock_pipe.to.return_value = cls.mock_pipe

        # Mock returned image from pipe call
        fake_img = Image.new("RGB", (512, 512), color="blue")
        mock_output = MagicMock()
        mock_output.images = [fake_img]
        cls.mock_pipe.return_value = mock_output

        cls.sd_patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=cls.mock_pipe)
        cls.sd_patcher.start()

        # Import app after patching
        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.sd_patcher.stop()

    def setUp(self):
        self.mock_pipe.reset_mock()
        fake_img = Image.new("RGB", (512, 512), color="blue")
        mock_output = MagicMock()
        mock_output.images = [fake_img]
        self.mock_pipe.return_value = mock_output
        self.mock_pipe.to.return_value = self.mock_pipe

    def test_generate_image_empty_prompt(self):
        img, status = self.app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)
        self.mock_pipe.assert_not_called()

    def test_generate_image_valid_prompt_runs_in_inference_mode(self):
        inference_mode_active = []

        def side_effect(*args, **kwargs):
            # Check if torch.is_inference_mode_enabled() is True inside pipe execution
            inference_mode_active.append(torch.is_inference_mode_enabled())
            fake_img = Image.new("RGB", (512, 512), color="red")
            output = MagicMock()
            output.images = [fake_img]
            return output

        self.mock_pipe.side_effect = side_effect

        img, status = self.app.generate_image(
            prompt="a glowing cyberpunk city",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)
        self.assertTrue(len(inference_mode_active) > 0)
        self.assertTrue(inference_mode_active[0], "Expected torch.inference_mode() to be active during generation")

    def test_generate_image_exception_handling(self):
        self.mock_pipe.side_effect = RuntimeError("CUDA out of memory")

        img, status = self.app.generate_image(
            prompt="a fantasy landscape",
            negative_prompt="",
            steps=25,
            guidance=7.0,
            width=512,
            height=512,
            seed=-1,
        )

        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", status)


if __name__ == "__main__":
    unittest.main()
