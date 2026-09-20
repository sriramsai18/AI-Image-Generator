## 2026-02-27 - PyTorch Inference Autograd Overhead in Diffusion Generation
**Learning:** Executing HuggingFace `diffusers` pipelines without `torch.inference_mode()` retains PyTorch autograd tracking state and version counters during inference, incurring avoidable CPU/GPU memory allocation overhead.
**Action:** Always wrap `pipe(...)` image generation invocations in `with torch.inference_mode():` to minimize overhead during inference.
