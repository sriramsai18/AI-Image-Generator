# Bolt's Journal

## 2024-05-20 - PyTorch Inference Mode & Memory Layout Optimization
**Learning:** In PyTorch/Diffusers inference pipelines, wrapping generation in `torch.inference_mode()` disables autograd tracking/overhead and enables C++ optimization, reducing latency by ~5-10% and memory consumption. Also, applying memory format channels_last on unet/vae improves GPU/CPU tensor throughput.
**Action:** Use `with torch.inference_mode():` during pipeline execution and set `to(memory_format=torch.channels_last)` where appropriate.
