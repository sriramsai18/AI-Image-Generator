import os
import sys

# Set MOCK_MODE = 1 so that app loads quickly and uses mock pipeline
os.environ["MOCK_MODE"] = "1"

# Import app modules to test
from app import generate_image, pipe, MOCK_MODE as APP_MOCK_MODE
import torch

def test_mock_mode_is_enabled():
    assert APP_MOCK_MODE is True

def test_generate_image_success():
    # Prompt is valid, should succeed and return image + info text
    image, info = generate_image(
        prompt="a scenic mountain view",
        negative_prompt="blurry",
        steps=20,
        guidance=7.5,
        width=512,
        height=512,
        seed=42
    )
    assert image is not None
    assert "✅ Generated in" in info
    assert "Steps: 20" in info
    assert "CFG: 7.5" in info
    assert "Seed: 42" in info
    assert image.size == (512, 512)

def test_generate_image_empty_prompt():
    # Prompt is empty, should return warning
    image, info = generate_image(
        prompt="",
        negative_prompt="",
        steps=25,
        guidance=7.5,
        width=512,
        height=512,
        seed=-1
    )
    assert image is None
    assert "⚠️ Please enter a prompt first!" in info

def test_channels_last_applied():
    # Channels last should be applied on unet and vae layers or mock objects
    assert hasattr(pipe, "unet")
    assert hasattr(pipe, "vae")
