# Bolt's Journal - Critical Learnings

## 2026-08-30 - Stable Diffusion Inference Optimization
**Learning:** Wrapping PyTorch pipeline generation in `torch.inference_mode()` disables autograd graph tracking and gradient state allocations, significantly reducing inference latency and memory overhead. Additionally, setting memory format to `channels_last` for UNet and VAE modules speeds up 2D Convolution operations.
**Action:** Always enable `torch.inference_mode()` for evaluation/generation loops and apply `channels_last` layout memory format to torch CNN components in Stable Diffusion pipelines.
