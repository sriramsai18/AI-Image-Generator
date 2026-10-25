## 2025-10-25 - Memory format optimization & inference context
**Learning:** Applying `channels_last` memory layout to Diffusers UNet/VAE models and using `torch.inference_mode()` instead of default execution contexts reduces memory overhead and improves tensor throughput during generative diffusion inference passes.
**Action:** Always format UNet and VAE layers with `torch.channels_last` upon pipeline initialization and enclose generation calls in `with torch.inference_mode():`.
