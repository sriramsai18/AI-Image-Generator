## 2026-03-31 - Stable Diffusion Model Optimization with Channels Last & Inference Mode
**Learning:** PyTorch UNet and VAE modules in diffusers pipelines spend significant time in 2D convolutions and autograd tracking overhead. Setting memory format to `torch.channels_last` optimizes memory layout for NHWC operations, while wrapping execution with `torch.inference_mode()` disables autograd overhead and tracking.
**Action:** Always convert UNet/VAE models to `channels_last` memory format and wrap pipeline inference in `with torch.inference_mode():` for PyTorch image generation pipelines.
