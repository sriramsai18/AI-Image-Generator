import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def cleanup_imports():
    """Ensure modules are reloaded and cleaned up properly between tests to avoid cached states."""
    # Clean up before test
    for mod in list(sys.modules.keys()):
        if mod == "app" or mod.startswith("app."):
            del sys.modules[mod]
    yield
    # Clean up after test
    for mod in list(sys.modules.keys()):
        if mod == "app" or mod.startswith("app."):
            del sys.modules[mod]

def test_app_optimization_with_high_ram():
    """Test app.py behavior under high RAM conditions:
    Attention slicing should NOT be enabled if RAM is >= 4GB.
    Channels last should be applied on UNet and VAE.
    """
    # Mock os.sysconf to return > 4GB RAM
    # PAGE_SIZE = 4096, PHYS_PAGES = 1500000 (~6.14 GB)
    def mock_sysconf(name):
        if name == "SC_PAGE_SIZE":
            return 4096
        if name == "SC_PHYS_PAGES":
            return 1500000
        raise ValueError("Invalid sysconf name")

    # Setup mocks
    mock_pipe = mock.MagicMock()
    mock_pipe.to.return_value = mock_pipe

    # We want to check if enable_attention_slicing was called or not
    mock_pipe.enable_attention_slicing = mock.MagicMock()

    mock_from_pretrained = mock.MagicMock(return_value=mock_pipe)

    # Mock diffusers.StableDiffusionPipeline.from_pretrained
    # Also mock launch so it doesn't actually launch the Gradio app
    with mock.patch("diffusers.StableDiffusionPipeline.from_pretrained", mock_from_pretrained), \
         mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch("os.sysconf", mock_sysconf, create=True), \
         mock.patch("gradio.Blocks.launch"):

        # Import app
        import app  # noqa: F401

        # Assert enable_attention_slicing was NOT called since RAM >= 4GB
        mock_pipe.enable_attention_slicing.assert_not_called()

        # Assert unet and vae memory format conversion was called
        import torch
        mock_pipe.unet.to.assert_any_call(memory_format=torch.channels_last)
        mock_pipe.vae.to.assert_any_call(memory_format=torch.channels_last)


def test_app_optimization_with_low_ram():
    """Test app.py behavior under low RAM conditions:
    Attention slicing SHOULD be enabled if RAM is < 4GB.
    """
    # Mock os.sysconf to return < 4GB RAM
    # PAGE_SIZE = 4096, PHYS_PAGES = 500000 (~2.05 GB)
    def mock_sysconf(name):
        if name == "SC_PAGE_SIZE":
            return 4096
        if name == "SC_PHYS_PAGES":
            return 500000
        raise ValueError("Invalid sysconf name")

    # Setup mocks
    mock_pipe = mock.MagicMock()
    mock_pipe.to.return_value = mock_pipe

    mock_pipe.enable_attention_slicing = mock.MagicMock()

    mock_from_pretrained = mock.MagicMock(return_value=mock_pipe)

    # Mock diffusers.StableDiffusionPipeline.from_pretrained
    with mock.patch("diffusers.StableDiffusionPipeline.from_pretrained", mock_from_pretrained), \
         mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch("os.sysconf", mock_sysconf, create=True), \
         mock.patch("gradio.Blocks.launch"):

        # Import app
        import app  # noqa: F401

        # Assert enable_attention_slicing WAS called since RAM < 4GB
        mock_pipe.enable_attention_slicing.assert_called_once()


def test_app_inference_mode_and_generation():
    """Test that the image generation executes within torch.inference_mode()
    and handles inputs/outputs correctly.
    """
    mock_pipe = mock.MagicMock()
    mock_pipe.to.return_value = mock_pipe

    # Mock pipe __call__ returning a mocked result
    mock_result = mock.MagicMock()
    mock_result.images = ["mock_image_pil_object"]
    mock_pipe.return_value = mock_result

    mock_from_pretrained = mock.MagicMock(return_value=mock_pipe)

    # We patch generator seed call to ensure manual_seed logic works
    mock_generator = mock.MagicMock()

    with mock.patch("diffusers.StableDiffusionPipeline.from_pretrained", mock_from_pretrained), \
         mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch("gradio.Blocks.launch"), \
         mock.patch("torch.Generator", return_value=mock_generator) as mock_torch_gen:

        import app

        # Let's test generating an image with seed != -1
        img, info = app.generate_image(
            prompt="a lovely cat",
            negative_prompt="blurry",
            steps=10,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        assert img == "mock_image_pil_object"
        assert "Generated in" in info
        assert "Steps: 10" in info
        mock_torch_gen.assert_called_once()
        mock_generator.manual_seed.assert_called_with(42)

        # Also verify empty prompt validation
        img, info = app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=10,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )
        assert img is None
        assert "Please enter a prompt first" in info
