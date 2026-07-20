import os
import sys
from PIL import Image

# Force MOCK_MODE for testing
os.environ["MOCK_MODE"] = "1"

# Import functions from app
from app import generate_image, reset_to_defaults

def test_reset_to_defaults():
    """Verify that reset_to_defaults returns the correct initial default state values."""
    defaults = reset_to_defaults()
    assert len(defaults) == 9
    assert defaults[0] == ""  # PROMPT
    assert defaults[1] == "blurry, ugly, distorted, low quality, watermark"  # NEGATIVE PROMPT
    assert defaults[2] == 25  # INFERENCE STEPS
    assert defaults[3] == 7.5  # CFG
    assert defaults[4] == 512  # WIDTH
    assert defaults[5] == 512  # HEIGHT
    assert defaults[6] == -1  # SEED
    assert defaults[7] is None  # IMAGE
    assert defaults[8] == ""  # STATUS

def test_generate_image_empty_prompt():
    """Verify that generate_image correctly flags an empty or whitespace-only prompt."""
    img, info = generate_image("   ", "blurry", 25, 7.5, 512, 512, -1)
    assert img is None
    assert "Please enter a prompt" in info

def test_generate_image_mock_mode():
    """Verify that generate_image creates a valid PIL image in mock mode."""
    img, info = generate_image("a cyberpunk neon cat", "blurry", 20, 8.0, 512, 256, 42)
    assert isinstance(img, Image.Image)
    assert img.size == (512, 256)
    assert "✅ Generated (MOCK)" in info
    assert "Steps: 20" in info
    assert "CFG: 8.0" in info
    assert "Seed: 42" in info
