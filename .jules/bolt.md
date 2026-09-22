## 2024-05-20 - PyTorch Inference Optimization with torch.inference_mode()
**Learning:** Wrapping PyTorch/Diffusers inference pipelines with `torch.inference_mode()` disables autograd tracking and version counter updates, reducing latency and peak memory usage compared to default execution or `torch.no_grad()`.
**Action:** Always wrap PyTorch pipeline inference calls in `with torch.inference_mode():` when running pure inference tasks.
