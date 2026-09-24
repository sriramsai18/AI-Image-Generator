## 2026-02-27 - Gradio Focus-Visible High-Contrast Indicators
**Learning:** Gradio's default theme and custom dark themes often override standard input/button outlines, rendering keyboard focus state invisible or barely visible for keyboard navigation users.
**Action:** Always declare explicit `.gradio-container` higher-specificity `:focus-visible` CSS rules (`outline: 2px solid <color> !important; outline-offset: 2px !important;`) on interactive controls (`textarea`, `input`, `button`, `a`) when customizing Gradio dark themes.
