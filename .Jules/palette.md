## 2026-09-23 - High-contrast focus indicators in custom dark themes
**Learning:** Gradio's internal CSS injects `outline: none !important;` on standard inputs and textareas, obscuring focus for keyboard navigation when custom dark themes are applied.
**Action:** Always use high specificity selectors (e.g., `.gradio-container button:focus-visible`) with `!important` to enforce high-contrast keyboard focus indicators.
