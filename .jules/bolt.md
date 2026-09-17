# Bolt's Journal - Performance Learnings

## 2025-05-20 - Disable Autograd for Stable Diffusion Inference
**Learning:** PyTorch by default tracks gradient metadata and version counters during forward passes unless explicitly disabled. In Hugging Face Diffusers pipeline execution across multiple denoising steps, wrapping inference with `torch.inference_mode()` eliminates autograd graph construction and tensor version tracking overhead, reducing memory allocations and inference time per step.
**Action:** Always wrap PyTorch model inference loops and pipeline calls in `with torch.inference_mode():` or decorate generation handlers with `@torch.inference_mode()`.
