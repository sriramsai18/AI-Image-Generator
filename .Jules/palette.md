## 2024-05-18 - High contrast focus-visible in Gradio custom dark theme
**Learning:** Gradio components inject internal CSS default rules with `!important` on inputs and buttons, which can obscure standard tag-level `:focus` or `:focus-visible` styling in dark themes.
**Action:** Always scope custom `:focus-visible` focus rules with `.gradio-container` and specify `!important` on `outline` and `box-shadow` properties to ensure strong contrast and keyboard navigation visibility.
