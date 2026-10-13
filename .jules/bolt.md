## 2026-03-30 - PyTorch Memory Layout and Inference Mode
**Learning:** Applying `channels_last` memory layout (`NHWC`) to 2D convolutional networks like Stable Diffusion UNet/VAE and wrapping model calls in `torch.inference_mode()` significantly reduces memory overhead and disables unnecessary autograd tracking.
**Action:** Always format UNet/VAE model tensors with `memory_format=torch.channels_last` and wrap pipeline inference execution with `with torch.inference_mode():`.
