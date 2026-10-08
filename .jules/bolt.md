## 2025-02-15 - Stable Diffusion Memory Formatting and Inference Mode

**Learning:** Formatting Stable Diffusion pipeline submodules (`unet` and `vae`) with `torch.channels_last` optimizes 2D spatial convolution layout in memory, yielding higher memory bandwidth utilization. Additionally, wrapping diffusion model inference in `torch.inference_mode()` disables autograd overhead and view tracking during generation.

**Action:** Always apply `memory_format=torch.channels_last` to UNet/VAE models in PyTorch diffusion pipelines and wrap generation execution in `torch.inference_mode()`.
