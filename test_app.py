from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_imports():
    # Remove 'app' from sys.modules before each test so we load it fresh
    import sys

    if "app" in sys.modules:
        del sys.modules["app"]
    yield


@pytest.fixture
def mock_pipeline():
    # Mock the StableDiffusionPipeline class and its pretrained loading
    with patch(
        "diffusers.StableDiffusionPipeline.from_pretrained"
    ) as mock_from_pretrained:
        mock_pipe = MagicMock()
        # Set chaining behavior on the mock pipeline
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.enable_attention_slicing.return_value = mock_pipe
        mock_from_pretrained.return_value = mock_pipe
        yield mock_from_pretrained, mock_pipe


def test_app_initialization(mock_pipeline):
    # Mock CPU/GPU configuration to keep it simple and fast
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("gradio.Blocks.launch"),
    ):
        import app

        # Assert that the pipeline was initialized correctly
        mock_from_pretrained, _ = mock_pipeline
        mock_from_pretrained.assert_called_once_with(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=app.torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Verify custom CSS focus rules are embedded
        assert "Accessibility Keyboard Focus Indicators" in app.css
        assert ".gradio-container button:focus-visible" in app.css
        assert "outline: 2px solid #e63946 !important;" in app.css

        # Verify that app components are constructed
        assert app.demo is not None


def test_generate_image_validation(mock_pipeline):
    _, mock_pipe = mock_pipeline

    # Set the mock return value for pipeline call before import
    mock_img = MagicMock()
    mock_pipe.return_value = MagicMock(images=[mock_img])

    # Mock CPU/GPU configuration to keep it simple and fast
    with patch("torch.cuda.is_available", return_value=False):
        import app

        # Case 1: Empty prompt validation
        image, status = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        assert image is None
        assert "Please enter a prompt first" in status

        # Case 2: Successful execution flow
        image, status = app.generate_image(
            "a cyberpunk city", "", 25, 7.5, 512, 512, 1234
        )
        assert image == mock_img
        assert "Generated in" in status
        assert "Steps: 25" in status
        assert "CFG: 7.5" in status
        assert "Seed: 1234" in status
