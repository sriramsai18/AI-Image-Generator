## 2025-09-03 - Inference Mode for Diffusion Pipeline
**Learning:** Wrapping Stable Diffusion model generation calls in `torch.inference_mode()` disables autograd overhead and view-tracking in PyTorch, accelerating forward pass speed and reducing VRAM/RAM overhead.
**Action:** Always wrap `pipe(...)` calls in `with torch.inference_mode():` during inference in PyTorch/Diffusers apps.
