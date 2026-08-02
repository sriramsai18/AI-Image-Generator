# Bolt's Journal - Critical Learnings

## 2026-03-01 - Avoid Blind Attention Slicing on CPU
**Learning:** `enable_attention_slicing` was previously enabled unconditionally on CPU. This introduces severe CPU overhead (up to ~230% slowdown) and should be avoided unless physical RAM is extremely constrained (<4GB). On systems with higher RAM (e.g. ~8GB), keeping it disabled improves CPU generation speeds.
**Action:** Always query system physical RAM dynamically using standard libraries and only enable attention slicing if RAM is strictly less than 4GB.
