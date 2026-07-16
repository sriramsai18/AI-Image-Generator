import os
import pytest
from PIL import Image

# Ensure mock mode is enabled before importing app
os.environ["MOCK_MODE"] = "1"

import app

def test_generate_image_success():
    """Verify that calling generate_image returns a valid PIL Image and a success status message."""
    prompt = "A high-performance cyberpunk racer, 4k resolution"
    negative_prompt = "low quality, blurry"
    steps = 25
    guidance = 7.5
    width = 512
    height = 512
    seed = 42

    image, status = app.generate_image(prompt, negative_prompt, steps, guidance, width, height, seed)

    # Assertions
    assert image is not None, "Image should not be None in mock mode"
    assert isinstance(image, Image.Image), "Returned image should be a PIL Image object"
    assert image.width == width, f"Expected width {width}, got {image.width}"
    assert image.height == height, f"Expected height {height}, got {image.height}"

    assert status is not None
    assert "✅ Generated in" in status
    assert "Steps: 25" in status
    assert "CFG: 7.5" in status
    assert "Seed: 42" in status

def test_generate_image_empty_prompt():
    """Verify that calling generate_image with an empty/whitespace prompt returns None and a warning message."""
    prompt = "   "
    negative_prompt = ""
    steps = 25
    guidance = 7.5
    width = 512
    height = 512
    seed = -1

    image, status = app.generate_image(prompt, negative_prompt, steps, guidance, width, height, seed)

    # Assertions
    assert image is None, "Image should be None for empty prompt"
    assert "⚠️ Please enter a prompt first!" in status

def test_get_system_ram_gb():
    """Verify get_system_ram_gb function returns a positive float or the fallback value."""
    ram = app.get_system_ram_gb()
    assert isinstance(ram, float)
    assert ram > 0.0
