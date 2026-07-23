import os
import pytest
from PIL import Image

# Ensure mock mode is enabled for the import/test
os.environ["MOCK_MODE"] = "1"

from app import generate_image, reset_defaults

def test_generate_image_mock():
    """Verify that generate_image works under MOCK_MODE and returns a valid image."""
    img, info = generate_image(
        prompt="a test cyberpunk landscape",
        negative_prompt="blurry",
        steps=25,
        guidance=7.5,
        width=512,
        height=512,
        seed=-1
    )

    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.width == 512
    assert img.height == 512
    assert "[MOCK]" in info
    assert "Steps: 25" in info
    assert "CFG: 7.5" in info

def test_generate_image_empty_prompt():
    """Verify that an empty prompt returns None and a warning message."""
    img, info = generate_image(
        prompt="  ",
        negative_prompt="blurry",
        steps=25,
        guidance=7.5,
        width=512,
        height=512,
        seed=-1
    )

    assert img is None
    assert "Please enter a prompt first" in info

def test_reset_defaults():
    """Verify that reset_defaults returns the correct initial default state."""
    defaults = reset_defaults()

    # Expected outputs: prompt, negative_prompt, steps, guidance, width, height, seed, output_image, info_text
    assert len(defaults) == 9
    assert defaults[0] == ""
    assert defaults[1] == "blurry, ugly, distorted, low quality, watermark"
    assert defaults[2] == 25
    assert defaults[3] == 7.5
    assert defaults[4] == 512
    assert defaults[5] == 512
    assert defaults[6] == -1
    assert defaults[7] is None
    assert defaults[8] == ""
