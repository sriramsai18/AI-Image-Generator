## 2026-02-16 - High-Contrast Focus States in Dark Gradio Interfaces
**Learning:** Gradio's default input and button styling can obscure standard focus indicators in dark/cyberpunk themes, impairing keyboard navigation accessibility. Specificity (`.gradio-container`) and `!important` declarations are required to override Gradio's internal styles for `:focus-visible`.
**Action:** Always scope `:focus-visible` rules under `.gradio-container` with explicit high-contrast outlines (`outline: 2px solid ... !important`) when applying dark custom CSS in Gradio apps.
