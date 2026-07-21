import os
import sys

# Ensure MOCK_MODE is enabled before importing app
os.environ["MOCK_MODE"] = "1"

import pytest
from PIL import Image
from app import reset_to_defaults, generate_image, MOCK_MODE

def test_mock_mode_is_enabled():
    assert MOCK_MODE is True
    assert os.environ.get("MOCK_MODE") == "1"

def test_reset_to_defaults():
    prompt, neg_prompt, steps, guidance, width, height, seed = reset_to_defaults()
    assert prompt == ""
    assert "blurry, ugly" in neg_prompt
    assert steps == 25
    assert guidance == 7.5
    assert width == 512
    assert height == 512
    assert seed == -1

def test_generate_image_empty_prompt():
    img, info = generate_image("   ", "neg", 25, 7.5, 512, 512, -1)
    assert img is None
    assert "⚠️ Please enter a prompt first!" in info

def test_generate_image_success():
    img, info = generate_image("A cool cyberpunk cat", "blurry", 25, 7.5, 512, 512, 12345)
    assert isinstance(img, Image.Image)
    assert img.width == 512
    assert img.height == 512
    assert "✅ Generated in" in info
    assert "Steps: 25" in info
    assert "CFG: 7.5" in info
    assert "Seed: 12345" in info
