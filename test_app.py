import sys
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestAppPerformance(unittest.TestCase):
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_app_optimization_and_generation(self, mock_cuda, mock_from_pretrained):
        # Setup mock pipeline and modules
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae
        mock_pipe.to.return_value = mock_pipe

        # Mock image output
        fake_image = Image.new("RGB", (512, 512), color="blue")
        mock_result = MagicMock()
        mock_result.images = [fake_image]
        mock_pipe.return_value = mock_result

        mock_from_pretrained.return_value = mock_pipe

        # Import app module dynamically
        if "app" in sys.modules:
            import importlib

            import app
            importlib.reload(app)
        else:
            import app

        # Verify channels_last memory format optimization was set
        mock_unet.to.assert_called_with(memory_format=app.torch.channels_last)
        mock_vae.to.assert_called_with(memory_format=app.torch.channels_last)

        # Test generate_image function
        image, info = app.generate_image(
            prompt="a beautiful scenery",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42,
        )

        self.assertIsNotNone(image)
        self.assertIn("Generated in", info)

        # Verify empty prompt handling
        empty_img, empty_info = app.generate_image(
            prompt="   ",
            negative_prompt="",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )
        self.assertIsNone(empty_img)
        self.assertIn("Please enter a prompt", empty_info)


if __name__ == "__main__":
    unittest.main()
