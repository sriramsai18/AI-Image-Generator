## 2026-02-27 - Stable Diffusion Memory Formatting and Inference Mode Optimization

**Learning:** Stable Diffusion pipeline optimization in PyTorch benefits significantly from applying 'channels_last' memory formatting on UNet and VAE layers, and wrapping inference calls with 'torch.inference_mode()'. This disables autograd tracking and version counter overhead while accelerating 2D convolution kernel operations.
**Action:** Always format UNet/VAE memory layout with `pipe.unet.to(memory_format=torch.channels_last)` and wrap diffusion pipeline calls in `with torch.inference_mode():` when serving PyTorch inference models.
