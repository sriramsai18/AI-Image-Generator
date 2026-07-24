import os
import sys

# Set mock mode before importing app to avoid loading the heavy model in unit tests
os.environ["MOCK_MODE"] = "1"

import pytest
from PIL import Image
import app

def test_mock_mode_active():
    """Verify that MOCK_MODE environment variable is recognized as active."""
    assert app.MOCK_MODE is True
    assert app.pipe is None

def test_generate_image_empty_prompt():
    """Verify that generate_image returns a warning message when prompt is empty."""
    img, status = app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
    assert img is None
    assert "Please enter a prompt first" in status

def test_generate_image_success_mock():
    """Verify that generate_image under MOCK_MODE returns a PIL image of the requested size."""
    width = 256
    height = 384
    img, status = app.generate_image("a futuristic cyberpunk city", "blurry", 20, 8.0, width, height, 42)

    assert isinstance(img, Image.Image)
    assert img.size == (width, height)
    assert "[MOCK]" in status
    assert "Steps: 20" in status
    assert "CFG: 8.0" in status
    assert "Seed: 42" in status

def test_gradio_demo_structure():
    """Verify that the Gradio app contains correct Blocks layout and components."""
    assert hasattr(app, "demo")
    assert isinstance(app.demo, app.gr.Blocks)

    # Check that key variables/components exist on the demo
    assert hasattr(app, "prompt")
    assert hasattr(app, "generate_btn")
    assert hasattr(app, "output_image")
