## 2026-10-03 - Torch Inference Mode & Channels Last Memory Layout for Diffusion Models
**Learning:** In PyTorch execution of diffusion models (e.g. Stable Diffusion v1.5), executing forward passes inside `torch.inference_mode()` disables autograd tracking overhead completely, while converting UNet and VAE layers to `torch.channels_last` optimizes 2D spatial convolution memory layout and Tensor Core access patterns.
**Action:** Always wrap non-training forward passes in `with torch.inference_mode():` and apply `memory_format=torch.channels_last` to UNet/VAE models upon pipeline setup.
