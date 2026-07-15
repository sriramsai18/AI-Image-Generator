# Bolt's Journal - Critical Learnings

## 2025-02-17 - Stable Diffusion CPU Optimization Bottlenecks
**Learning:**
1. `pipe.enable_attention_slicing()` is a known memory-saving technique but causes up to ~230% CPU overhead slowdown on standard CPU environments with >4GB RAM. It should only be enabled when RAM is extremely constrained (<4GB).
2. PyTorch's `channels_last` memory format improves memory locality on UNet and VAE layers, yielding faster inference times on both CPU and GPU.
3. Wrapping Stable Diffusion pipeline execution with `torch.inference_mode()` avoids extra tensor version tracking overhead and optimizes execution speed.
4. Implementing a `MOCK_MODE` environment variable allows fast offline verification and saves multi-gigabyte downloads during routine development and frontend test runs.

**Action:** Ensure CPU-bound pipelines check available RAM before enabling attention slicing, apply `channels_last` on critical layers, and execute with `torch.inference_mode()`. Implement offline mock pipeline fallback via `MOCK_MODE`.
