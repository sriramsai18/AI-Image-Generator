## 2025-08-25 - PyTorch Inference Optimization with `inference_mode` and `channels_last`
**Learning:** Applying `@torch.inference_mode()` disables autograd overhead and tensor reference tracking during diffusion model execution. Concurrently, setting `torch.channels_last` memory format on UNet and VAE layers optimizes memory access patterns for 2D spatial convolution operations without breaking compatibility.
**Action:** Always wrap model inference handlers with `@torch.inference_mode()` and format convolutional vision layers with `channels_last` for PyTorch pipeline speedups.
