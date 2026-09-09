## 2026-03-09 - Stable Diffusion Pipeline Inference Optimization
**Learning:** Wrapping Stable Diffusion `pipe(...)` calls with `torch.inference_mode()` disables autograd tracking overhead and reduces memory allocations during generation. Additionally, setting `channels_last` memory format on UNet/VAE models improves convolution performance on compatible runtimes.
**Action:** Always wrap PyTorch diffusion model inference calls in `torch.inference_mode()` and format UNet/VAE layers to `torch.channels_last` for optimal execution speed.
