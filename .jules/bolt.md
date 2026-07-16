# Bolt's Journal

## 2025-01-20 - [Optimizing Stable Diffusion v1.5 Performance]
**Learning:**
1. Enabling attention slicing on CPU (using `enable_attention_slicing()`) can introduce a severe CPU overhead of up to ~230% slowdown when system memory is not extremely constrained (>=4GB). This should only be used as a fallback under extremely low RAM (<4GB).
2. Applying 'channels_last' memory format layout (`to(memory_format=torch.channels_last)`) on UNet and VAE layers leverages Tensor Cores and optimized vector instructions (like AVX) to speed up 2D convolutional networks significantly on both CPU and CUDA.
3. Wrapping Stable Diffusion inference with `torch.inference_mode()` instead of standard context or no context bypasses autograd tracking overhead entirely, producing a measurable speedup in runtime and memory consumption.
4. Setting a lightweight, offline mockup pipeline using `MOCK_MODE=1` environment variable avoids heavy model downloads and lets us quickly test UI functionality and performance-optimized logic flow safely.

**Action:**
1. Check system RAM programmatically using Python's standard `os` library (`SC_PAGE_SIZE` * `SC_PHYS_PAGES`). Only call `enable_attention_slicing()` on CPU if total RAM is under 4GB.
2. Apply `to(memory_format=torch.channels_last)` format specifically to `pipe.unet` and `pipe.vae` after model instantiation.
3. Wrap SD inference generation calls in `with torch.inference_mode():`.
