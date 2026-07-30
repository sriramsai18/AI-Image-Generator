## 2025-06-03 - Attention Slicing Overhead on CPU
**Learning:** Attention slicing (`enable_attention_slicing`) is designed to reduce memory usage at the cost of speed, but on CPU it introduces massive overhead (up to ~230% slowdown). It should only be enabled when system RAM is extremely constrained (< 4GB).
**Action:** Programmatically check system RAM before enabling attention slicing on CPU, and avoid it on machines with sufficient RAM.

## 2025-06-03 - Channels Last and Inference Mode Speedup
**Learning:** Stable Diffusion pipeline inference can be optimized by formatting UNet and VAE weights to `channels_last` (which aligns memory format with tensor operations) and by wrapping generation within a `torch.inference_mode()` context manager (which disables gradient tracking with less overhead than `torch.no_grad()`).
**Action:** Apply `to(memory_format=torch.channels_last)` to UNet and VAE, and wrap pipeline calls in `with torch.inference_mode():`.
