## 2026-02-20 - High-Contrast Focus-Visible Styles in Gradio Custom Themes
**Learning:** Gradio internal component styles often override default browser focus indicators with `outline: none !important;` on textareas and inputs, making keyboard navigation difficult in dark/neon custom themes.
**Action:** Always add high-specificity `.gradio-container *:focus-visible` rules with `!important` on `outline`, `outline-offset`, and `box-shadow` to guarantee accessible focus indicators for keyboard navigation.
