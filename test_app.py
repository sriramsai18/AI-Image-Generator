import os
import pytest
from PIL import Image

# Force MOCK_MODE for tests
os.environ["MOCK_MODE"] = "1"

import app

def test_mock_mode_active():
    """Verify that mock mode is correctly detected and pipeline is not loaded."""
    assert app.MOCK_MODE is True
    assert app.pipe is None

def test_generate_image_empty_prompt():
    """Verify behavior of generate_image with empty prompt."""
    image, message = app.generate_image("", "ugly", 25, 7.5, 512, 512, -1)
    assert image is None
    assert "Please enter a prompt first" in message

def test_generate_image_valid_mock():
    """Verify behavior of generate_image mock generation."""
    image, message = app.generate_image("test prompt", "ugly", 25, 7.5, 256, 256, 42)
    assert isinstance(image, Image.Image)
    assert image.size == (256, 256)
    assert "MOCK MODE" in message
    assert "Steps: 25" in message
    assert "CFG: 7.5" in message
    assert "Seed: 42" in message

def test_get_system_ram_gb():
    """Verify system RAM helper returns a float greater than 0."""
    ram = app.get_system_ram_gb()
    assert isinstance(ram, float)
    assert ram > 0.0

def test_gradio_elements():
    """Verify that key Gradio components and elements are loaded correctly."""
    assert app.demo is not None
    assert app.demo.title == "Text2Image — Sriram"
