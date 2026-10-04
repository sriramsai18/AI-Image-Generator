## 2025-05-18 - Stable Diffusion Inference Optimization with Inference Mode and Channels Last

**Learning:** Running PyTorch Stable Diffusion pipelines without explicit `torch.inference_mode()` context causes unnecessary autograd tracking and memory allocation overhead. Additionally, default contiguous NCHW memory layout in UNet and VAE layers is sub-optimal for 2D convolution operations compared to `torch.channels_last` (NHWC layout).

**Action:** Always wrap `pipe(...)` generation calls with `with torch.inference_mode():` and convert `pipe.unet` and `pipe.vae` tensors to `torch.channels_last` memory format during pipeline setup.
