# Bolt's Journal

## 2025-02-15 - Stable Diffusion CPU/GPU Inference Optimization
**Learning:** Attention slicing (`enable_attention_slicing`) significantly degrades CPU inference speed (causing up to ~230% slowdown) by introducing heavy slicing overhead. It should only be used when RAM is extremely constrained (<4GB). Furthermore, Stable Diffusion pipelines can be optimized greatly on both CPU and GPU by applying the `channels_last` memory format on UNet/VAE convolutional layers, and by executing pipeline calls inside PyTorch's `torch.inference_mode()` context manager rather than `torch.no_grad()` to completely eliminate view-tracking and version counter overhead.
**Action:** Always conditionally enable attention slicing based on physical RAM availability (queried using `os.sysconf` on Unix or default fallback), use `channels_last` layout on 2D convolutional model components, and wrap model generation within `torch.inference_mode()` for maximum speed.
