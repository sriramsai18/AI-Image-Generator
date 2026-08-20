## 2026-08-20 - Optimizing Diffusers Inference with channels_last and inference_mode
**Learning:** In PyTorch Stable Diffusion pipelines, converting UNet and VAE layers to `torch.channels_last` memory layout optimizes 2D spatial tensor convolutions, while wrapping inference calls in `torch.inference_mode()` disables autograd tracking and reduces memory allocation overhead.
**Action:** Always apply `pipe.unet.to(memory_format=torch.channels_last)` and `with torch.inference_mode():` during Stable Diffusion inference passes.
