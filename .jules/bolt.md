## 2025-05-18 - Stable Diffusion PyTorch Inference Optimizations

**Learning:** In PyTorch and Diffusers pipelines, `enable_attention_slicing()` trades computational throughput for lower memory consumption, causing up to 2x-3x slowdown on CPU/GPU when VRAM/RAM is not constrained. Performance can be significantly enhanced by:
1. Omitting attention slicing unless operating under extreme memory constraints (<4GB).
2. Converting UNet layers to `torch.channels_last` memory format (NHWC) for faster 2D convolution kernel execution on CPU and CUDA.
3. Wrapping model evaluation calls in `torch.inference_mode()` (or `torch.no_grad()`) to eliminate autograd graph construction overhead.

**Action:** Always check memory requirements before enabling memory-saving features like attention slicing. Apply `channels_last` to UNet models and wrap evaluation with `torch.inference_mode()`.
