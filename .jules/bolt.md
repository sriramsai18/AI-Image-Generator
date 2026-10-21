## 2026-02-27 - PyTorch Inference Mode for Diffusion Pipelines
**Learning:** Running PyTorch Stable Diffusion pipeline generation calls without `torch.inference_mode()` context manager causes autograd tracking, version counter updates, and tensor graph construction overhead during multi-step denoising iterations.
**Action:** Always wrap PyTorch diffusion model inference calls in `with torch.inference_mode():` to eliminate autograd graph overhead and reduce runtime/memory consumption.
