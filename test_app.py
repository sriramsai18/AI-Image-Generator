import os
import pytest
from PIL import Image

# Force MOCK_MODE before importing app to avoid loading the heavy Stable Diffusion model
os.environ["MOCK_MODE"] = "1"

import app

def test_generate_image_empty_prompt():
    image, info = app.generate_image("", "ugly", 25, 7.5, 512, 512, -1)
    assert image is None
    assert "Please enter a prompt first" in info

def test_generate_image_whitespace_prompt():
    image, info = app.generate_image("   ", "ugly", 25, 7.5, 512, 512, -1)
    assert image is None
    assert "Please enter a prompt first" in info

def test_generate_image_success():
    prompt = "A high-tech sci-fi cyberpunk visualizer, dramatic neon lighting"
    image, info = app.generate_image(prompt, "ugly", 25, 7.5, 512, 512, -1)

    # Verify image returned is a valid PIL Image of correct dimensions
    assert isinstance(image, Image.Image)
    assert image.width == 512
    assert image.height == 512

    # Verify the status text contains expected metadata
    assert "Generated in" in info
    assert "Steps: 25" in info
    assert "CFG: 7.5" in info
    assert "Seed: random" in info

def test_generate_image_custom_seed_and_dims():
    prompt = "Golden wheat field at sunset"
    image, info = app.generate_image(prompt, "ugly", 15, 8.0, 256, 384, 42)

    assert isinstance(image, Image.Image)
    assert image.width == 256
    assert image.height == 384
    assert "Seed: 42" in info
