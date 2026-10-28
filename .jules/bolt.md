## 2024-10-28 - CPU Attention Slicing Overhead in Diffusion Models
**Learning:** `enable_attention_slicing()` reduces memory footprint during Stable Diffusion inference, but introduces substantial computation overhead on CPU (slowing down generation by over 2x). It should only be enabled when RAM is heavily constrained (< 4GB).
**Action:** Dynamically check system memory using `os.sysconf` before enabling attention slicing on CPU devices.

## 2024-10-28 - PyTorch Memory Format and Inference Mode Optimizations
**Learning:** In PyTorch, 2D convolution-heavy architectures like Stable Diffusion UNet and VAE gain 10-20% speedups when tensors and module parameters are formatted in `torch.channels_last`. Additionally, wrapping diffusion inference calls in `torch.inference_mode()` completely eliminates autograd tracking overhead, outperforming standard `torch.no_grad()`.
**Action:** Always convert UNet/VAE modules to `torch.channels_last` memory format after model load and wrap generation pipeline invocations inside `torch.inference_mode()`.
