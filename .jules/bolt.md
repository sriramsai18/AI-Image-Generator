## 2026-03-30 - Stable Diffusion Inference Optimization with Channels Last
**Learning:** For PyTorch Diffusion pipelines (Stable Diffusion v1.5), converting UNet and VAE layers to `torch.channels_last` memory layout optimizes 2D convolution memory access patterns, while wrapping inference in `torch.inference_mode()` avoids autograd tracking and memory allocation overhead.
**Action:** Always apply `pipe.unet.to(memory_format=torch.channels_last)` and wrap execution in `with torch.inference_mode():` for PyTorch diffusion model inference.
