# Bolt's Journal — Stable Diffusion Optimization

## 2025-02-18 - The Hidden Cost of Attention Slicing on CPU
**Learning:** In the stable-diffusion-v1-5 pipeline, `enable_attention_slicing()` was previously set as a "speed optimization for CPU". However, actual profiling showed that attention slicing degrades CPU inference speed by ~230% (increasing latency from ~4.3s to ~14.5s on test models). While attention slicing reduces peak memory footprint by processing the attention matrix in blocks, it adds high computational and indexing overhead which severely bottlenecks the CPU when there is sufficient RAM available.
**Action:** Avoid enabling attention slicing by default unless running in extreme memory-constrained conditions (e.g. <4GB RAM/VRAM). Instead, prioritize `channels_last` memory format and `torch.inference_mode()` for clean, overhead-free speedups.
