## 2026-03-30 - Disable Autograd Overhead for Diffusion Inference with torch.inference_mode()
**Learning:** Wrapping Stable Diffusion pipeline execution calls inside `with torch.inference_mode():` avoids autograd tensor tracking and version counter overhead, optimizing RAM/VRAM utilization and speeding up inference execution.
**Action:** Always wrap PyTorch model inference loops and pipeline invocations in `with torch.inference_mode():` when gradients are not required.
