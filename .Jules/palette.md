## 2024-08-18 - Gradio Keyboard Focus Specificity
**Learning:** Gradio default styles inject low-level focus resets. Standard `:focus` rules may get overridden unless scoped with `.gradio-container` and using `!important` on `outline` and `box-shadow`.
**Action:** Always scope custom keyboard focus accessibility styles with `.gradio-container` and `!important` declarations in Gradio apps.
