## 2026-02-18 - Stable Diffusion Inference Optimization with inference_mode and channels_last

**Learning:** Wrapping Stable Diffusion generation calls with `torch.inference_mode()` disables autograd overhead and view tracking, yielding faster inference execution and reduced memory footprint compared to default evaluation. In addition, setting `channels_last` (NHWC) memory format on UNet and VAE layers leverages PyTorch's optimized C++ NHWC convolution kernels, speeding up tensor operations. Crucially, calling `enable_attention_slicing()` on CPU severely hurts throughput (up to ~230% slowdown) due to slicing overhead and should be avoided unless RAM is extremely constrained (<4GB).

**Action:** Always wrap diffusion generation passes in `with torch.inference_mode():`, apply `channels_last` memory layout to UNet/VAE models, and avoid CPU attention slicing unless low-memory conditions strictly require it.
