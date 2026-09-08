## 2025-05-18 - Inference Acceleration with torch.inference_mode()

**Learning:** Wrapping PyTorch/Diffusers model forward passes and pipeline invocations in `torch.inference_mode()` disables autograd tracking and optimizes tensor computations, reducing memory allocation overhead and speeding up model inference compared to standard execution or `torch.no_grad()`.

**Action:** Always wrap model generation calls in `with torch.inference_mode():` during production inference routines.
