# Bolt's Journal - Critical Learnings

## 2025-07-31 - CPU Attention Slicing Overhead & Inference Optimizations
**Learning:** `enable_attention_slicing()` in Stable Diffusion introduces severe CPU overhead (up to ~230% slowdown) and should be avoided on CPU unless system RAM is extremely constrained (<4GB). For optimal Stable Diffusion performance, use `channels_last` memory layout on convolutional models (like UNet and VAE) and wrap generation in `torch.inference_mode()` to skip gradient tracking overhead.
**Action:** Detect system RAM size using standard library (`os.sysconf`) before enabling attention slicing on CPU, format UNet and VAE layers using `channels_last` layout, and wrap inference blocks in `torch.inference_mode()` context.
