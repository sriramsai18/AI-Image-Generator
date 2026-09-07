## 2025-09-07 - Overriding Gradio default focus styles
**Learning:** Gradio's default UI component styling applies default outline and shadow rules that can obscure high-contrast keyboard focus indicators in dark/neon themed interfaces.
**Action:** Always scope `:focus-visible` rules under `.gradio-container` using `!important` on `outline` and `box-shadow` properties to ensure accessible keyboard navigation.
