# Bolt's Journal — Performance Optimizations

## 2025-02-18 - CPU Attention Slicing Overhead
**Learning:** Attention slicing (`enable_attention_slicing`) on CPU introduces up to ~230% severe computational overhead. It should only be used when RAM is extremely constrained (< 4GB). Since the system has ~7.77 GB of RAM, disabling attention slicing yields a dramatic speedup.
**Action:** Avoid calling `enable_attention_slicing()` on CPU unless system RAM is < 4GB.
