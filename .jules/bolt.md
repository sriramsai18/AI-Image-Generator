## 2026-03-30 - Stable Diffusion Inference Optimizations
**Learning:** Applying `channels_last` memory layout to `pipe.unet` and wrapping generation with `torch.inference_mode()` reduces autograd memory allocations and speeds up 2D convolution execution in Diffusers pipelines.
**Action:** Always format UNet memory layout with `channels_last` and use `torch.inference_mode()` for PyTorch model inference loops.
