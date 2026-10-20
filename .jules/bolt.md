## 2026-03-31 - Stable Diffusion Inference Optimization with torch.inference_mode and channels_last
**Learning:** Wrapping Stable Diffusion pipeline execution with `torch.inference_mode()` reduces PyTorch runtime overhead by turning off view tracking and autograd metadata. Converting UNet and VAE modules to `torch.channels_last` optimizes 2D spatial convolution memory layout.
**Action:** Always wrap PyTorch diffusion inference calls in `torch.inference_mode()` and set memory format to `channels_last` on UNet and VAE modules.
