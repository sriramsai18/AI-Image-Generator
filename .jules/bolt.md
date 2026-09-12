## 2026-02-27 - Disable Autograd Tracking for Inference
**Learning:** PyTorch pipelines executed without explicit `torch.inference_mode()` track tensor versioning and graph metadata unnecessarily during generation passes.
**Action:** Always wrap model generation calls in `with torch.inference_mode():` to minimize runtime memory allocations and overhead.
