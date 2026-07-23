## 2025-02-18 - [Stable Diffusion Performance Optimizations & CPU Attention Slicing Pitfalls]
**Learning:**
- Wrapping Stable Diffusion inference with `torch.inference_mode()` and applying `channels_last` memory format to UNet and VAE layers maximizes performance and minimizes peak memory usage.
- Unconditionally calling `enable_attention_slicing()` on CPU causes severe overhead (up to ~230% slowdown). It should only be used when RAM is extremely constrained (<4GB).
- System RAM can be safely retrieved on Unix platforms without third-party dependencies (like `psutil`) by calling `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')`.

**Action:**
- Ensure `channels_last` and `torch.inference_mode()` are standard optimization defaults for Stable Diffusion pipelines.
- Verify system RAM size dynamically before enabling memory saving options like attention slicing, and skip attention slicing if RAM is 4GB or more.
