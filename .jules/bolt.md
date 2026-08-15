## 2025-08-15 - Stable Diffusion PyTorch Inference Optimizations
**Learning:** Stable Diffusion pipeline execution in diffusers benefits significantly from `torch.inference_mode()` context wrapper (eliminates autograd graph overhead) and `channels_last` (NHWC) memory format on `unet` and `vae` modules for faster 2D convolutions.
**Action:** Always wrap `pipe(...)` invocations in `torch.inference_mode()` and convert CNN weights (`unet`, `vae`) to `channels_last` memory layout when deploying PyTorch diffusion models.
