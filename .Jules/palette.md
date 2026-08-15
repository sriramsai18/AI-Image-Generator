## 2026-03-30 - Gradio High-Contrast Focus Visible Overrides
**Learning:** Gradio 3.50.2 applies default input outline rules that can hide focus rings in custom dark themes. To ensure high keyboard navigation accessibility, explicit `:focus-visible` selectors scoped under `.gradio-container` with `!important` on `outline` and `box-shadow` are necessary.
**Action:** Always scope custom CSS focus-visible rules with `.gradio-container` and use `!important` to reliably override Gradio's internal styles for interactive elements (button, input, textarea, a).
