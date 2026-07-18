# Bolt's Journal

⚡ A performance-obsessed agent's log of critical codebase-specific performance insights.

## 2025-07-17 - Attention Slicing Overhead & Memory Layout Formatting
**Learning:** In PyTorch/Stable Diffusion pipelines, `pipe.enable_attention_slicing()` introduces severe CPU overhead (up to ~230% slowdown) when the system has sufficient memory (>=4GB RAM) and runs on CPU. Optimizing performance on CPU and GPU requires conditionally avoiding attention slicing if system RAM is sufficient, while applying `channels_last` memory formatting to UNet and VAE modules and wrapping inference inside `torch.inference_mode()` context.
**Action:** Conditionally enable attention slicing only when physical RAM is strictly less than 4GB. Always format UNet/VAE to `channels_last` memory format and wrap pipeline execution in `torch.inference_mode()` for high-throughput, low-overhead generation.
