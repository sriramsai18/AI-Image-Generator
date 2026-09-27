## 2026-09-27 - Gradio Focus-Visible High-Contrast Focus Indicators
**Learning:** Gradio's internal CSS components inject `outline: none !important` on form inputs, textareas, and buttons, which suppresses default browser focus rings and leaves keyboard-navigating users without visual feedback in custom dark themes.
**Action:** Always scope custom keyboard focus rules with `.gradio-container` and declare `outline` and `box-shadow` with `!important` on `:focus-visible` to guarantee keyboard accessibility across buttons, inputs, textareas, and links.
