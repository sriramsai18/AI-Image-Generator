## 2025-05-18 - UNet Memory Layout and Inference Mode Optimizations
**Learning:** Applying `channels_last` memory format to PyTorch UNet models and wrapping Stable Diffusion pipeline execution in `torch.inference_mode()` significantly reduces memory overhead and inference latency during image generation without changing tensor precision or outputs.
**Action:** Always wrap model generation calls with `with torch.inference_mode():` and convert 2D convolution modules (like UNet/VAE) to `memory_format=torch.channels_last` when optimizing diffusers pipelines.
