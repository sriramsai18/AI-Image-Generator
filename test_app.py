import os
import sys
import pytest
from PIL import Image

# Ensure MOCK_MODE is 1 for testing
os.environ["MOCK_MODE"] = "1"

# Import app modules after setting MOCK_MODE
import app

def test_mock_pipeline_initialization():
    """Verify that MockPipeline is correctly initialized and has the appropriate layer mocks."""
    assert app.MOCK_MODE is True
    assert hasattr(app, "pipe")
    assert app.pipe is not None
    assert hasattr(app.pipe, "unet")
    assert hasattr(app.pipe, "vae")

    # Test unet and vae .to() mock methods
    assert app.pipe.unet.to("cuda") == app.pipe.unet
    assert app.pipe.vae.to("cpu") == app.pipe.vae

def test_empty_prompt_handling():
    """Check that calling generate_image with an empty prompt returns a warning message and no image."""
    img, info = app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
    assert img is None
    assert "Please enter a prompt first" in info

    img, info = app.generate_image("   ", "blurry", 25, 7.5, 512, 512, -1)
    assert img is None
    assert "Please enter a prompt first" in info

def test_successful_image_generation():
    """Check that calling generate_image with valid inputs generates a PIL Image and returns correct info."""
    prompt = "A majestic glowing space nebula, high-contrast neon styling"
    negative_prompt = "blurry, low resolution"
    steps = 30
    guidance = 8.5
    width = 512
    height = 512
    seed = 42

    img, info = app.generate_image(prompt, negative_prompt, steps, guidance, width, height, seed)

    # Verify image properties
    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.size == (width, height)

    # Verify status info elements
    assert "Generated in" in info
    assert "Steps: 30" in info
    assert "CFG: 8.5" in info
    assert "Seed: 42" in info

def test_random_seed_handling():
    """Verify that using seed = -1 returns 'random' in the status string."""
    img, info = app.generate_image("neon tree", "", 25, 7.5, 512, 512, -1)
    assert img is not None
    assert "Seed: random" in info
