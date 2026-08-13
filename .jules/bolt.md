## 2026-03-01 - Attention Slicing Overhead on CPU
**Learning:** Calling `enable_attention_slicing` on StableDiffusionPipeline under CPU inference introduces massive CPU overhead (up to ~230% slowdown) and should be avoided if system memory is >= 4GB.
**Action:** Dynamically check system RAM on POSIX systems via `os.sysconf` and only enable attention slicing if RAM is < 4GB.

## 2026-03-01 - Channels Last Memory Optimization
**Learning:** Formatting UNet and VAE layers of the Stable Diffusion pipeline in PyTorch to `channels_last` (using `to(memory_format=torch.channels_last)`) significantly improves performance on convolutional layers.
**Action:** Convert UNet and VAE components to channels_last format right after pipeline instantiation.

## 2026-03-01 - PyTorch Inference Mode Context
**Learning:** Wrapping inference with `torch.inference_mode()` context manager offers better performance gains and lower memory overhead compared to `torch.no_grad()`.
**Action:** Use `with torch.inference_mode():` around the model call in the generation function.
