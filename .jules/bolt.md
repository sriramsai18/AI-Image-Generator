## 2026-03-30 - Torch Inference Mode and Channels Last Optimization
**Learning:** Wrapping Stable Diffusion generation calls with `torch.inference_mode()` disables autograd overhead and tensor version counter tracking during inference. In addition, converting the UNet module memory format to `channels_last` optimizes 2D convolution execution for PyTorch CNNs.
**Action:** Always wrap PyTorch diffusion model inference in `with torch.inference_mode():` and convert 2D CNN backbones (UNet) to `channels_last` memory format.
