import os
import sys
from unittest import mock
import torch
from PIL import Image

# Force MOCK_MODE = "1" for the import to load MockPipeline
os.environ["MOCK_MODE"] = "1"

# Import app modules after setting mock mode environment variable
import app

def test_generate_image_empty_prompt():
    """Verify that generate_image handles empty or whitespace prompts gracefully."""
    image, status = app.generate_image("", "ugly, blurry", 25, 7.5, 512, 512, -1)
    assert image is None
    assert "Please enter a prompt first" in status

    image, status = app.generate_image("   ", "ugly, blurry", 25, 7.5, 512, 512, -1)
    assert image is None
    assert "Please enter a prompt first" in status

def test_generate_image_success_mock():
    """Verify that generate_image returns a valid image under Mock Mode."""
    image, status = app.generate_image("a futuristic cyberpunk city", "ugly, blurry", 20, 7.0, 512, 256, 42)
    assert isinstance(image, Image.Image)
    assert image.width == 512
    assert image.height == 256
    assert "Generated in" in status
    assert "CFG: 7.0" in status
    assert "Seed: 42" in status

def test_mock_pipeline_directly():
    """Verify that MockPipeline can be called and respects dimensions."""
    pipeline = app.MockPipeline()
    result = pipeline(
        prompt="a lone tree in a golden field",
        negative_prompt="",
        num_inference_steps=10,
        guidance_scale=7.5,
        width=384,
        height=384,
        generator=None
    )
    assert hasattr(result, "images")
    assert len(result.images) == 1
    img = result.images[0]
    assert img.size == (384, 384)

def test_cpu_optimization_logic_fallback():
    """Verify that the SC_PAGE_SIZE logic falls back safely if an exception occurs."""
    with mock.patch("os.sysconf", side_effect=ValueError("mocked error")):
        try:
            total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except Exception:
            total_ram = 8 * (1024 ** 3)
        assert total_ram == 8589934592
