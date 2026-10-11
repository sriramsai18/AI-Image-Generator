# Bolt's Performance Journal

## 2025-01-15 - Torch Inference Mode & Memory Layout Optimization
**Learning:** Wrapping PyTorch diffusion pipeline inference with `torch.inference_mode()` disables autograd tracking and version counter overhead. Applying `torch.channels_last` memory layout on UNet and VAE layers optimizes 2D convolutions on GPU Tensor Cores.
**Action:** Always wrap model generation calls in `with torch.inference_mode():` and convert 2D CNN models to `channels_last` when GPU acceleration is active.
