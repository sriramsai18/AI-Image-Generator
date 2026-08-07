import sys
import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image

class TestStableDiffusionOptimization(unittest.TestCase):

    def setUp(self):
        # We need to clean up app from sys.modules to ensure reload works correctly
        if "app" in sys.modules:
            del sys.modules["app"]

    def create_mock_pipeline(self):
        # Create a mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe

        # Mock UNet and VAE components
        mock_unet = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.unet = mock_unet
        mock_pipe.vae = mock_vae

        # Mock the __call__ method to return a valid result
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (256, 256))]
        mock_pipe.return_value = mock_result

        return mock_pipe

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_pipeline_on_cpu_with_high_ram(self, mock_sysconf, mock_cuda_avail, mock_from_pretrained):
        # Mock high RAM scenario: e.g., 8 GB
        # SC_PAGE_SIZE = 4096, SC_PHYS_PAGES = 2097152 => 8 GB
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            elif name == "SC_PHYS_PAGES":
                return 2097152
            raise ValueError()
        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = self.create_mock_pipeline()
        mock_from_pretrained.return_value = mock_pipe

        # Import/load app module
        import app  # noqa: F401

        # Verify channels_last format is applied to unet and vae
        mock_pipe.unet.to.assert_any_call(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_any_call(memory_format=torch.channels_last)

        # Verify enable_attention_slicing is NOT called when RAM is high (>=4GB)
        mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    def test_pipeline_on_cpu_with_low_ram(self, mock_sysconf, mock_cuda_avail, mock_from_pretrained):
        # Mock low RAM scenario: e.g., 2 GB
        # SC_PAGE_SIZE = 4096, SC_PHYS_PAGES = 524288 => 2 GB
        def sysconf_side_effect(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            elif name == "SC_PHYS_PAGES":
                return 524288
            raise ValueError()
        mock_sysconf.side_effect = sysconf_side_effect

        mock_pipe = self.create_mock_pipeline()
        mock_from_pretrained.return_value = mock_pipe

        # Import/load app module
        import app  # noqa: F401

        # Verify enable_attention_slicing is indeed called when RAM is low (<4GB)
        mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_generate_image_inference_mode(self, mock_cuda_avail, mock_from_pretrained):
        mock_pipe = self.create_mock_pipeline()
        mock_from_pretrained.return_value = mock_pipe

        inference_mode_active = [False]

        # Define side effect to verify torch.is_inference_mode_enabled() during pipeline execution
        def mock_pipeline_call(*args, **kwargs):
            inference_mode_active[0] = torch.is_inference_mode_enabled()
            mock_result = MagicMock()
            mock_result.images = [Image.new("RGB", (256, 256))]
            return mock_result

        mock_pipe.side_effect = mock_pipeline_call

        # Import/load app module
        import app

        # Call generate_image
        image, info = app.generate_image(
            prompt="A beautiful landscape",
            negative_prompt="blurry",
            steps=10,
            guidance=7.5,
            width=256,
            height=256,
            seed=42
        )

        # Ensure image generated successfully and info is returned
        self.assertIsNotNone(image)
        self.assertIn("✅ Generated in", info)

        # Assert that pipeline was executed while torch.is_inference_mode_enabled() was True
        self.assertTrue(inference_mode_active[0], "Pipeline call was not executed inside torch.inference_mode")

if __name__ == "__main__":
    unittest.main()
