import os
os.environ["MOCK_MODE"] = "1"

from app import reset_inputs, generate_image

def test_reset_inputs():
    expected = ("", "blurry, ugly, distorted, low quality, watermark", 25, 7.5, 512, 512, -1)
    assert reset_inputs() == expected

def test_generate_image_mock():
    image, status = generate_image(
        prompt="test prompt",
        negative_prompt="blurry",
        steps=10,
        guidance=7.5,
        width=256,
        height=256,
        seed=42
    )
    assert image is not None
    assert "Generated" in status
    assert image.size == (256, 256)
