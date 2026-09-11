## 2026-03-01 - Avoid unconditional attention slicing on CPU and leverage inference mode

**Learning:** `enable_attention_slicing()` in Diffusers is a memory-saving technique that computes attention sequentially in chunks. On CPU inference, this introduces substantial Python loop and slicing overheads, causing up to ~230% slowdown in image generation. Additionally, running model calls without `torch.inference_mode()` incurs unnecessary autograd tracking overhead and memory allocations.
**Action:** Only enable attention slicing when system RAM is severely constrained (<4GB). Always wrap pipeline inference calls in `with torch.inference_mode():` and convert UNet memory format to `channels_last` for optimal tensor convolution speed.
