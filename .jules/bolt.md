## 2026-08-23 - UNet Channels Last & Inference Mode Optimization
**Learning:** Formatting UNet model tensors in `channels_last` memory format speeds up 2D convolutions in PyTorch, while wrapping inference with `torch.inference_mode()` disables autograd overhead and tracking.
**Action:** Always format UNet layers with `memory_format=torch.channels_last` on pipeline initialization and use `torch.inference_mode()` during diffusion model execution.
