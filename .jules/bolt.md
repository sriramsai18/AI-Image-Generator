# Bolt's Journal - Critical Learnings Only

## 2025-07-24 - Stable Diffusion Optimization on CPU/GPU
**Learning:**
- PyTorch attention slicing (`enable_attention_slicing()`) on CPUs introduces massive computational overhead, causing up to a ~230% slowdown in image generation times. It should be avoided on systems that are not extremely RAM-constrained (< 4GB).
- Converting UNet and VAE layers of Hugging Face diffusers to the `channels_last` (NHWC) memory format leverages highly optimized PyTorch vector instruction paths (like AVX/AMX on CPU and Tensor Cores on GPU), speeding up convolutional operations.
- Executing model inference inside `torch.inference_mode()` instead of standard context or `torch.no_grad()` removes all autograd overhead and accelerates execution times on both CPU and GPU.

**Action:**
- Only enable attention slicing if running on a CPU with extremely constrained RAM (< 4GB).
- Explicitly convert UNet and VAE modules to `channels_last` memory format during pipeline initialization.
- Wrap the generation loop / model call with `with torch.inference_mode():`.
