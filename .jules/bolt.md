## 2026-03-30 - Stable Diffusion Inference Optimization with Inference Mode and Memory Formatting

**Learning:** Stable Diffusion pipeline optimization in this repository benefits from applying 'channels_last' memory formatting on UNet and VAE layers, and wrapping inference with `torch.inference_mode()`. `torch.inference_mode()` disables autograd tracking completely, eliminating version counter updates and tensor tracking overhead during forward passes. Furthermore, formatting UNet and VAE layers using `torch.channels_last` optimizes 2D convolution memory access patterns for PyTorch's native C++/CUDA operators.

**Action:** Wrap PyTorch pipeline calls in `with torch.inference_mode():` and convert UNet/VAE models to `torch.channels_last` memory layout upon pipeline initialization.
