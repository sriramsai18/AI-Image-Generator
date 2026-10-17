# Bolt's Journal - Critical Learnings

## 2026-02-27 - Stable Diffusion Inference Optimizations: Channels-Last Memory Format and Inference Mode
**Learning:** Running Diffusers `StableDiffusionPipeline` inference without `torch.inference_mode()` incurs unnecessary PyTorch autograd graph tracking and version counter overhead. Furthermore, converting UNet and VAE module memory formats to `channels_last` (`torch.channels_last` / NHWC layout) significantly speeds up 2D spatial convolutions across modern PyTorch engines.
**Action:** Always wrap PyTorch model inference in `torch.inference_mode()` and format UNet/VAE memory layouts with `torch.channels_last` upon pipeline initialization.
