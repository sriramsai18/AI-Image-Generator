## 2025-05-18 - High-contrast Keyboard Focus Indicators in Dark Gradio Themes
**Learning:** Gradio's internal UI components apply default outline resets that can obscure keyboard focus states in dark or custom-themed web applications.
**Action:** Always declare explicit, high-contrast `:focus-visible` styles scoped to `.gradio-container` with `!important` on `outline` and `box-shadow` properties to ensure keyboard navigation accessibility across interactive elements.
