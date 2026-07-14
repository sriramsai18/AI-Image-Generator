# Bolt's Journal

## 2025-07-14 - [Memory Format and Inference Mode Optimization]
**Learning:** Stable Diffusion pipeline optimization benefits significantly from applying 'channels_last' memory formatting to the UNet and VAE layers, and wrapping inference in 'torch.inference_mode()'. Also, `enable_attention_slicing` introduces severe CPU overhead (up to ~230% slowdown) and should be avoided on CPU unless RAM is extremely constrained (<4GB).
**Action:** Apply 'channels_last' formatting to UNet and VAE modules, wrap generator inference block in `torch.inference_mode()`, and avoid `enable_attention_slicing` on CPU when system RAM is >= 4GB.
