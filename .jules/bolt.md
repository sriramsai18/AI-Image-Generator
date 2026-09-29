## 2026-03-29 - Stable Diffusion Pipeline Optimizations with torch.inference_mode and channels_last

**Learning:** Wrapping PyTorch Stable Diffusion pipeline execution with `torch.inference_mode()` disables autograd history and view tracking, reducing runtime overhead during model generation. Additionally, setting `memory_format=torch.channels_last` on UNet and VAE modules optimizes memory layout for 2D convolutions in PyTorch.

**Action:** Always wrap model generation in `torch.inference_mode()` and convert CNN/diffusion UNet models to `channels_last` memory format for optimal inference throughput.
