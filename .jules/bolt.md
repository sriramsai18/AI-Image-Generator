## 2025-08-06 - Stable Diffusion Memory Format and CPU Attention Slicing Optimizations
**Learning:** Stable Diffusion pipeline inference can be significantly accelerated by applying three optimizations:
1. Formatting UNet and VAE layers with `channels_last` memory format (`to(memory_format=torch.channels_last)`) to improve memory access locality.
2. Wrapping inference execution inside a `torch.inference_mode()` context manager to eliminate gradient tracking and autograd overhead.
3. Avoiding CPU attention slicing (`enable_attention_slicing()`) unless RAM is extremely limited (< 4GB), as attention slicing can cause severe CPU generation overhead (up to ~230% slowdown) when memory is not constrained.
**Action:** Check system RAM constraints using standard library interfaces (e.g., `os.sysconf`) before activating memory-saving CPU optimizations, and always use memory formats and inference context managers appropriate for PyTorch models.
