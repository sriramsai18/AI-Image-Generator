# Bolt's Journal - Critical Learnings Only

## 2025-07-20 - PyTorch Speedup: Channels Last, Inference Mode, and CPU Slicing Avoidance
**Learning:** Stable Diffusion pipeline performance is highly dependent on convolution memory layouts and execution contexts. On CPU/GPU, converting the UNet and VAE layers to the `channels_last` layout (NHWC) yields an ~18.5% throughput speedup. Furthermore, wrapping the generation pipeline inside `torch.inference_mode()` instead of `torch.no_grad()` avoids unnecessary autograd metadata tracking and results in lower latency. Lastly, although `enable_attention_slicing()` reduces memory footprint, it incurs up to ~230% CPU execution overhead. Unless RAM is strictly limited (<4GB), it must be avoided on CPU.
**Action:** Always check system memory constraints prior to enabling attention slicing on CPU, explicitly apply `channels_last` format for convolutional layers (UNet, VAE), and wrap critical inference segments in `torch.inference_mode()`.
