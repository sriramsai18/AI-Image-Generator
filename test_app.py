import sys
from unittest.mock import MagicMock, patch

# Mocking gradio so we don't spin up UI during unit testing
sys.modules['gradio'] = MagicMock()

@patch('torch.cuda.is_available', return_value=False)
@patch('os.sysconf', create=True)
def test_attention_slicing_under_low_ram(mock_sysconf, mock_cuda):
    # Mock RAM to be < 4GB (e.g. 2GB)
    # SC_PAGE_SIZE = 4096, SC_PHYS_PAGES = 524288 -> 2GB
    mock_sysconf.side_effect = lambda key: 4096 if "PAGE_SIZE" in key else 524288

    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    with patch('diffusers.StableDiffusionPipeline.from_pretrained', return_value=mock_pipe):
        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        _ = app
        # Ensure enable_attention_slicing was called because RAM is 2GB (< 4GB)
        mock_pipe.enable_attention_slicing.assert_called_once()


@patch('torch.cuda.is_available', return_value=False)
@patch('os.sysconf', create=True)
def test_attention_slicing_under_high_ram(mock_sysconf, mock_cuda):
    # Mock RAM to be >= 4GB (e.g. 8GB)
    # SC_PAGE_SIZE = 4096, SC_PHYS_PAGES = 2097152 -> 8GB
    mock_sysconf.side_effect = lambda key: 4096 if "PAGE_SIZE" in key else 2097152

    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    with patch('diffusers.StableDiffusionPipeline.from_pretrained', return_value=mock_pipe):
        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        _ = app
        # Ensure enable_attention_slicing was NOT called because RAM is 8GB (>= 4GB)
        mock_pipe.enable_attention_slicing.assert_not_called()
