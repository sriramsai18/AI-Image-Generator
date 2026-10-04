## 2025-02-14 - Gradio Custom Styling Focus States
**Learning:** Gradio's internal components often override standard CSS focus outlines on input and textarea tags. Adding `:focus-visible` selectors with `.gradio-container` scope and `!important` flags ensures keyboard navigation indicators remain high-contrast and accessible across dark theme components.
**Action:** Always scope `:focus-visible` selectors to `.gradio-container` with `!important` on `outline` and `box-shadow` when customizing Gradio UI themes.
