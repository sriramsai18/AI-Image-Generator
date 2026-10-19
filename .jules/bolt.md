# Bolt's Performance Journal

## 2024-10-19 - CPU Attention Slicing & Memory Layout Optimization
**Learning:** `enable_attention_slicing()` on CPU introduces severe loop/slicing overhead (~230% slowdown) unless system RAM is extremely constrained (<4GB). Applying `channels_last` memory format to UNet/VAE and wrapping inference in `torch.inference_mode()` reduces tensor allocation and 2D convolution latency significantly.
**Action:** Only enable attention slicing when system RAM is <4GB, apply `channels_last` to UNet/VAE, and always wrap inference in `torch.inference_mode()`.
