import os
from PIL import Image

# Set MOCK_MODE environment variable to 1 before importing app
os.environ["MOCK_MODE"] = "1"
import app

def test_mock_mode_image_generation():
    """Test that image generation in Mock Mode works correctly and returns a PIL image of requested dimensions."""
    prompt = "a majestic snow-capped mountain"
    negative_prompt = "cartoon"
    steps = 25
    guidance = 7.5
    width = 512
    height = 512
    seed = -1

    image, info = app.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed
    )

    # Verification
    assert isinstance(image, Image.Image), "The output must be a PIL Image instance"
    assert image.size == (width, height), f"The output image size {image.size} must match the requested resolution {(width, height)}"
    assert "Generated" in info, "The status info should indicate successful generation"
    assert "Steps: 25" in info, "The steps in the info must match the inputs"

def test_empty_prompt_error():
    """Test that empty or whitespace prompts are rejected with a warning message."""
    image, info = app.generate_image(
        prompt="   ",
        negative_prompt="blurry",
        steps=25,
        guidance=7.5,
        width=512,
        height=512,
        seed=-1
    )

    assert image is None, "Image should be None for empty prompts"
    assert "Please enter a prompt first" in info, "Status should contain warning message"

def test_seed_handling():
    """Test that setting a valid seed is reflected in the generation info."""
    prompt = "cyberpunk city"
    negative_prompt = ""
    steps = 15
    guidance = 8.0
    width = 384
    height = 384
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
    assert "Seed: 42" in info, f"Seed info must match, but got: {info}"
