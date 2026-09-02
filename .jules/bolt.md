## 2026-09-02 - Stable Diffusion UNet Optimization & Memory Layout

**Learning:** Stable Diffusion pipeline generation speed benefits significantly from applying PyTorch `channels_last` (NHWC) memory format on UNet and VAE layers, and wrapping inference with `torch.inference_mode()`. Enabling attention slicing on CPU introduces severe sequential iteration overhead (up to ~230% slowdown) and should only be applied as a fallback when total system RAM is extremely constrained (<4GB).

**Action:** Prefer `memory_format=torch.channels_last` for diffusers UNet/VAE and `torch.inference_mode()` for pure inference, and check system RAM before enabling attention slicing on CPU.
