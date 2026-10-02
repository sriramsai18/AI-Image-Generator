## 2024-10-02 - Diffusers UNet Memory Layout and Inference Mode Optimization
**Learning:** Applying `channels_last` memory format to UNet coupled with `torch.inference_mode()` disables autograd overhead and optimizes 2D convolution operations without modifying pipeline API or outputs.
**Action:** Always format UNet parameters to `channels_last` and wrap pipeline generation calls with `with torch.inference_mode():` in PyTorch generation tasks.
