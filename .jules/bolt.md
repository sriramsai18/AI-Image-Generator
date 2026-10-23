## 2025-05-18 - Stable Diffusion Memory Format & Inference Mode
**Learning:** Applying `memory_format=torch.channels_last` to UNet/VAE modules and executing inference inside `torch.inference_mode()` reduces PyTorch runtime overhead and speeds up 2D convolution execution.
**Action:** Always format 2D CNN-based diffusion models to `channels_last` memory layout and disable autograd/view-tracking with `torch.inference_mode()` during inference.
