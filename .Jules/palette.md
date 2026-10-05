## 2025-02-23 - Gradio Focus-Visible High Specificity Requirement
**Learning:** Gradio's internal CSS overrides default browser focus rings on input, textarea, and button elements. To guarantee accessible keyboard navigation on dark custom themes, `:focus-visible` styles must be prefixed with `.gradio-container` and use `!important` on `outline` and `box-shadow`.
**Action:** Always scope custom keyboard focus rings under `.gradio-container [element]:focus-visible` with `!important` flags when enhancing dark-themed Gradio applications.
