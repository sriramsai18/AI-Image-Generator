# Bolt's Journal — Critical Learnings

This journal is a record of critical architectural performance learnings, unexpected behavior of optimizations, or rejected changes.

## 2025-02-18 - Stable Diffusion Performance Optimizations
**Learning:**
1. Unconditional `enable_attention_slicing()` on CPU introduces up to ~230% slowdown (CPU overhead) and should be avoided unless RAM is extremely constrained (<4GB).
2. Wrapping generation inference with `torch.inference_mode()` instead of relying on default behavior avoids autograd tracking overhead.
3. Applying `channels_last` memory format to UNet and VAE layers optimizes memory layout for faster convolutions.

**Action:**
1. Dynamically enable attention slicing on CPU only if system RAM is < 4GB.
2. Use `@torch.inference_mode()` on the generation function.
3. Convert UNet and VAE weights to `torch.channels_last` memory layout.
