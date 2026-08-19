## 2026-02-28 - PyTorch Inference Optimization with Inference Mode and Channels Last Memory Format
**Learning:** In Stable Diffusion pipelines (and PyTorch CNNs in general), executing model inference without autograd tracking (`torch.inference_mode()`) and formatting UNet/VAE memory layout as `channels_last` significantly reduces memory overhead and accelerates execution of 2D convolutions.
**Action:** Always wrap `pipe(...)` image generation calls in `torch.inference_mode()` and convert UNet/VAE model weights to `channels_last` memory format when loading CUDA pipelines.
