# Bolt's Journal

## 2025-07-27 - Stable Diffusion CPU and Pipeline Optimization
**Learning:** Enabling attention slicing unconditionally on CPU introduces up to a ~230% slowdown when system RAM is not extremely constrained (>=4GB). Also, Stable Diffusion UNet and VAE layers benefit significantly from `channels_last` memory formatting, and wrapping inference inside `torch.inference_mode()` reduces overhead and memory consumption compared to default execution or standard `no_grad()`.
**Action:** Avoid calling `enable_attention_slicing()` on CPU unless system RAM is <4GB, which can be checked using Python's standard library `os.sysconf`. Optimize the pipeline with `channels_last` memory format and run inference within `torch.inference_mode()`.
