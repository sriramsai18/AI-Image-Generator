## 2025-02-23 - Scoped `:focus-visible` Rules in Gradio Applications
**Learning:** Gradio's internal default CSS uses `!important` on reset rules for inputs and buttons, which overrides base tag `:focus-visible` styles on dark themes.
**Action:** Always scope custom `:focus-visible` rules under `.gradio-container` and apply `!important` to both `outline` and `box-shadow` properties.
