## 2025-10-21 - High-Contrast Focus Indicators in Gradio Dark Themes
**Learning:** Custom dark/neon themes in Gradio can obscure standard browser outline rings for interactive elements (inputs, textareas, buttons, and links), making keyboard navigation difficult for visually impaired users or keyboard-only navigators.
**Action:** Always include high-contrast `:focus-visible` rules prefixed with `.gradio-container` using `!important` on `outline` and `box-shadow` properties to ensure focus indicators remain distinct and prominent across all interactive components.
