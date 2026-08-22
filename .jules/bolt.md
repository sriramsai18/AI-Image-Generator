## 2025-05-20 - Avoid Attention Slicing on CPU when RAM is Sufficient
**Learning:** Attention slicing (`enable_attention_slicing()`) reduces peak memory usage but introduces significant CPU overhead (up to ~230% slowdown) during inference. It should only be enabled when RAM is heavily constrained (<4GB).
**Action:** Query system RAM using `os.sysconf` before enabling attention slicing on CPU devices.
