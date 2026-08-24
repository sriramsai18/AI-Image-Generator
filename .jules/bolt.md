## 2024-08-24 - Avoid attention slicing on non-constrained RAM systems and leverage channels_last & inference_mode

**Learning:** `enable_attention_slicing()` in `diffusers.StableDiffusionPipeline` saves RAM at the cost of splitting attention computation into iterative step slices, creating significant loop overhead (~22% CPU slowdown per inference step). Replacing attention slicing with `memory_format=torch.channels_last` on UNet and VAE layers along with wrapping inference in `torch.inference_mode()` yields a ~22% latency reduction (e.g. 79.1s down to 61.7s for 5 steps on CPU) on systems with sufficient RAM.

**Action:** Only enable `enable_attention_slicing()` when RAM is severely constrained (<4GB). Otherwise, optimize Stable Diffusion CPU/GPU pipelines using `channels_last` and `torch.inference_mode()`.
