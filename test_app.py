import os
import pytest

# Force mock mode for all tests
os.environ["MOCK_MODE"] = "1"

from app import generate_image, demo, pipe, MOCK_MODE
from PIL import Image

def test_mock_mode_is_enabled():
    assert MOCK_MODE is True
    assert pipe.__class__.__name__ == "MockPipeline"

def test_generate_image_empty_prompt():
    image, info = generate_image("", "ugly", 25, 7.5, 512, 512, -1)
    assert image is None
    assert "Please enter a prompt" in info

def test_generate_image_valid_prompt():
    image, info = generate_image("a cute kitten", "blurry", 10, 7.5, 256, 256, 42)
    assert isinstance(image, Image.Image)
    assert image.size == (256, 256)
    assert "✅ Generated in" in info
    assert "Steps: 10" in info
    assert "Seed: 42" in info

def test_gradio_elements():
    # Verify that the Gradio demo block loads with correct title and elements
    assert demo.title == "Text2Image — Sriram"
    # Ensure standard components exist in the demo app
    assert len(demo.blocks) > 0
