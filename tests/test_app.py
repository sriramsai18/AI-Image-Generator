# ruff: noqa: E402
import sys
from unittest.mock import MagicMock, patch

# To avoid loading the actual heavy pipeline during tests, mock StableDiffusionPipeline
mock_pipe = MagicMock()
mock_pipe_instance = MagicMock()
mock_result = MagicMock()

# Mock generated image
from PIL import Image
mock_img = Image.new("RGB", (128, 128), color="red")
mock_result.images = [mock_img]
mock_pipe_instance.return_value = mock_result
mock_pipe.from_pretrained.return_value = mock_pipe_instance

sys.modules["diffusers"] = MagicMock()
sys.modules["diffusers"].StableDiffusionPipeline = mock_pipe

# We should mock torch.cuda.is_available() to False to test CPU path
with patch("torch.cuda.is_available", return_value=False), \
     patch("os.sysconf", create=True) as mock_sysconf:
    # Set mock memory to 8GB (more than 4GB)
    # page size * num pages
    mock_sysconf.side_effect = lambda name: 4096 if name == "SC_PAGE_SIZE" else (2 * 1024 * 1024 if name == "SC_PHYS_PAGES" else 0)

    # Now import generate_image from app
    import app
    from app import generate_image

def test_generate_image_success():
    # Setup mock pipeline behavior
    app.pipe = MagicMock()
    mock_res = MagicMock()
    mock_res.images = [mock_img]
    app.pipe.return_value = mock_res

    img, info = generate_image(
        prompt="a test prompt",
        negative_prompt="",
        steps=25,
        guidance=7.5,
        width=512,
        height=512,
        seed=123
    )
    assert img is not None
    assert "✅ Generated in" in info
    assert "Steps: 25" in info

def test_generate_image_empty_prompt():
    img, info = generate_image(
        prompt="",
        negative_prompt="",
        steps=25,
        guidance=7.5,
        width=512,
        height=512,
        seed=-1
    )
    assert img is None
    assert "Please enter a prompt first" in info
