import os
import pytest
from PIL import Image

# Ensure mock mode is enabled before importing app
os.environ["MOCK_MODE"] = "1"
import app

def test_mock_mode_initialization():
    """Verify that the application successfully initializes in MOCK_MODE."""
    assert app.MOCK_MODE is True
    assert app.pipe is not None
    # Verify mock pipeline has required methods and layers
    assert hasattr(app.pipe, "unet")
    assert hasattr(app.pipe, "vae")
    assert hasattr(app.pipe, "enable_attention_slicing")
    assert hasattr(app.pipe, "disable_attention_slicing")

def test_generate_image_success():
    """Verify that generate_image returns a valid PIL Image and correct info metadata."""
    prompt = "a majestic space nebula, highly detailed, 8k"
    negative_prompt = "blurry, dark"
    steps = 30
    guidance = 8.5
    width = 512
    height = 512
    seed = 42

    image, info = app.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed
    )

    assert isinstance(image, Image.Image)
    assert image.width == width
    assert image.height == height
    assert "✅ Generated in" in info
    assert f"Steps: {steps}" in info
    assert f"CFG: {guidance}" in info
    assert f"Seed: {seed}" in info

def test_generate_image_empty_prompt():
    """Verify that an empty or whitespace-only prompt is rejected with a warning."""
    image, info = app.generate_image(
        prompt="   ",
        negative_prompt="",
        steps=25,
        guidance=7.5,
        width=512,
        height=512,
        seed=-1
    )

    assert image is None
    assert "⚠️ Please enter a prompt first!" in info

def test_generate_image_random_seed():
    """Verify that passing seed=-1 yields a randomized seed status text."""
    image, info = app.generate_image(
        prompt="neon cyber dragon",
        negative_prompt="",
        steps=25,
        guidance=7.5,
        width=512,
        height=512,
        seed=-1
    )

    assert isinstance(image, Image.Image)
    assert "Seed: random" in info

def test_generate_image_custom_dimensions():
    """Verify that the generated image dimensions match custom settings."""
    image, info = app.generate_image(
        prompt="minimalist architecture",
        negative_prompt="cluttered",
        steps=20,
        guidance=7.0,
        width=256,
        height=384,
        seed=123
    )

    assert isinstance(image, Image.Image)
    assert image.width == 256
    assert image.height == 384
