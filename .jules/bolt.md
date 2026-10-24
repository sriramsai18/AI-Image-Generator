## 2024-10-24 - Stable Diffusion Memory Formatting & Inference Mode
**Learning:** Applying `channels_last` memory format to PyTorch UNet/VAE models and wrapping generation calls in `torch.inference_mode()` significantly reduces tensor operation overhead during inference.
**Action:** Always format 2D CNN pipeline modules to `channels_last` and use `torch.inference_mode()` for PyTorch generation routines.
