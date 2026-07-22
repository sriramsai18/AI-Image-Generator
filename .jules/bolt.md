# Bolt's Journal ⚡

## 2025-06-25 - Stable Diffusion CPU Optimization and Memory Formats
**Learning:** For CPU-bound Stable Diffusion inference, enabling attention slicing (`enable_attention_slicing()`) when RAM is sufficient (>= 4GB) incurs a significant processing overhead (~11% slower in micro-benchmarks). Disabling attention slicing and opting for the channels_last memory format (`torch.channels_last`) on the UNet and VAE layers, combined with `torch.inference_mode()`, increases overall throughput by ~18.3%.
**Action:** Dynamically check the system's physical memory before enabling attention slicing, format UNet/VAE to use `channels_last`, and wrap generation within a `torch.inference_mode()` context.
