# Bolt's Performance Journal

## 2026-09-01 - Optimizing Stable Diffusion Pipeline Memory and Execution
**Learning:** PyTorch UNet operations run significantly faster when using `channels_last` memory format (`pipe.unet.to(memory_format=torch.channels_last)`), and wrapping pipeline execution in `torch.inference_mode()` eliminates PyTorch autograd overhead during inference. Additionally, `enable_attention_slicing()` introduces up to 200%+ CPU execution overhead when memory is not constrained.
**Action:** Always format UNet memory layout to `channels_last`, wrap diffusion generation in `torch.inference_mode()`, and avoid attention slicing unless system RAM is under 4GB.
