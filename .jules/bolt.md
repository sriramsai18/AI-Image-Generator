# Bolt's Journal - Critical Learnings Only

## 2025-07-25 - Stable Diffusion Memory Formatting & CPU Slowdown
**Learning:**
1. Enabling attention slicing (`enable_attention_slicing`) on CPU causes a severe performance regression (~230% slowdown) when abundant system RAM is available. It should only be enabled on extremely constrained systems (< 4GB RAM).
2. Applying `channels_last` memory layout to the UNet and VAE layers of the Stable Diffusion pipeline improves inference throughput.
3. Wrapping model inference in `torch.inference_mode()` instead of `torch.no_grad()` provides extra performance benefits by bypassing autograd tracking entirely.
4. Setting up a robust `MOCK_MODE` allows full offline development and frontend layout verification without downloading gigabytes of weights.

**Action:**
- Implement system RAM checking before enabling attention slicing.
- Apply `channels_last` conversion to PyTorch model components.
- Wrap pipeline execution in `torch.inference_mode()`.
- Ensure mock mode is always integrated into UI apps for fast development cycles.
