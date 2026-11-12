## 2026-04-18 - Stable Diffusion PyTorch Inference Optimizations
**Learning:** PyTorch default inference in diffusers pipelines can incur hidden autograd bookkeeping unless wrapped in `torch.inference_mode()`. Furthermore, 2D convolutional networks (UNet and VAE) achieve lower latency and higher tensor core / vectorization utilization when converted to `channels_last` memory format.
**Action:** Always wrap pipeline inference calls in `with torch.inference_mode():` and configure UNet and VAE layers with `to(memory_format=torch.channels_last)`.
