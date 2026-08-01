from unittest.mock import MagicMock, patch
import sys

# Mock the StableDiffusionPipeline class and its pretrained loading
mock_from_pretrained_patch = patch(
    "diffusers.StableDiffusionPipeline.from_pretrained"
)
mock_from_pretrained = mock_from_pretrained_patch.start()
mock_pipe = MagicMock()
mock_pipe.to.return_value = mock_pipe
mock_pipe.enable_attention_slicing.return_value = mock_pipe
mock_from_pretrained.return_value = mock_pipe

# Mock torch.cuda.is_available
is_cuda_patch = patch("torch.cuda.is_available", return_value=False)
is_cuda_patch.start()

# Now import the app and launch it
import app

if __name__ == "__main__":
    app.demo.launch(server_name="127.0.0.1", server_port=7860, prevent_thread_lock=False)
