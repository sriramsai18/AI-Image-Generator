## 2025-05-10 - Channels-Last Memory Format and Inference Mode for Stable Diffusion
**Learning:** In PyTorch Stable Diffusion pipelines, converting UNet and VAE module memory layout to `torch.channels_last` optimizes 2D convolution operations, and wrapping generation calls in `torch.inference_mode()` disables autograd tracking and intermediate memory allocations for faster inference.
**Action:** Always format UNet/VAE models to `channels_last` memory layout and wrap inference pipeline calls with `with torch.inference_mode():`.
