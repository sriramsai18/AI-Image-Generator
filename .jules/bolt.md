## 2025-08-10 - CPU Slicing Overhead & Tensor Formatting

**Learning:** Enabling `enable_attention_slicing` when sufficient system memory is available (>4GB RAM) on CPU causes an unnecessary and severe execution slowdown (up to ~230% overhead) due to sub-optimal sequential attention block processing. Additionally, PyTorch inference operations on Stable Diffusion run faster and with less memory when the UNet and VAE modules use `channels_last` layout formatting combined with `torch.inference_mode()` context wrapping.

**Action:** Always verify system resources (RAM size) using `os.sysconf` before unconditionally applying CPU memory optimizations like attention slicing. Apply memory format optimizations (`channels_last`) on the core heavy layers of the model, and restrict gradient tracking using `torch.inference_mode()` during inference.
