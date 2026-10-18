## 2026-03-30 - PyTorch Inference Optimization with channels_last and inference_mode

**Learning:** Standard PyTorch pipelines incur unnecessary autograd/tensor-tracking overhead during inference unless wrapped in `torch.inference_mode()`. Additionally, Stable Diffusion UNet and VAE 2D convolutions execute faster when model parameters use the NHWC memory format (`torch.channels_last`).
**Action:** Always wrap PyTorch pipeline calls in `with torch.inference_mode():` and convert UNet/VAE memory layout to `torch.channels_last` for inference workloads.
