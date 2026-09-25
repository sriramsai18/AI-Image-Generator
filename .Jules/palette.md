## 2026-09-25 - Gradio High-Contrast Focus Indicators
**Learning:** Gradio components often strip or suppress default focus outlines. In dark-themed web interfaces, default browser focus rings blend into dark backgrounds, making keyboard navigation difficult for screen reader and keyboard users.
**Action:** Always scope custom `:focus-visible` CSS rules under `.gradio-container` with higher specificity and `!important` flags for `outline` and `box-shadow` across all interactive elements (`button`, `input`, `textarea`, `select`, `a`, `[role="button"]`).
