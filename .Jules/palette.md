## 2024-09-06 - Gradio High-Contrast Keyboard Focus Indicators
**Learning:** Custom dark-themed Gradio interfaces often obscure or remove default browser focus outlines on buttons, text inputs, and links, making keyboard navigation difficult or non-functional for accessibility.
**Action:** Always inspect custom CSS in Gradio apps and add explicit `:focus-visible` rules scoped under `.gradio-container` with `!important` on `outline` and `box-shadow`.
