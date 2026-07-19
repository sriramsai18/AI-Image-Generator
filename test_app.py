import sys
from unittest.mock import MagicMock

# 1. Mock diffusers module before importing app to avoid loading the actual model
mock_pipe_instance = MagicMock()
mock_pipe_instance.to.return_value = mock_pipe_instance

# Mock the result of pipe(...) call
mock_result = MagicMock()
mock_image = MagicMock()
mock_result.images = [mock_image]
mock_pipe_instance.return_value = mock_result

mock_sd_pipeline = MagicMock()
mock_sd_pipeline.from_pretrained.return_value = mock_pipe_instance

sys.modules['diffusers'] = MagicMock()
import diffusers
diffusers.StableDiffusionPipeline = mock_sd_pipeline

# 2. Import app
import app

def test_reset_inputs():
    """Test that resetting the app's inputs returns standard default values."""
    results = app.reset_inputs()
    assert len(results) == 9
    prompt, neg_prompt, steps, guidance, width, height, seed, output_image, info_text = results

    assert prompt == ""
    assert neg_prompt == "blurry, ugly, distorted, low quality, watermark"
    assert steps == 25
    assert guidance == 7.5
    assert width == 512
    assert height == 512
    assert seed == -1
    assert output_image is None
    assert "Defaults restored" in info_text

def test_generate_image_empty_prompt():
    """Test that an empty prompt is handled with a warning message."""
    image, info = app.generate_image("   ", "ugly", 25, 7.5, 512, 512, -1)
    assert image is None
    assert "Please enter a prompt" in info

def test_generate_image_success():
    """Test successful image generation calls pipeline and returns image."""
    image, info = app.generate_image("a cute cat", "ugly", 20, 8.0, 512, 512, 42)
    assert image == mock_image
    assert "Generated in" in info
    assert "Steps: 20" in info
    assert "CFG: 8.0" in info
    assert "Seed: 42" in info
    mock_pipe_instance.assert_called()
