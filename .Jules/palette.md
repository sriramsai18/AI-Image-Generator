## 2025-08-30 - Gradio Custom CSS Focus-Visible Indicators
**Learning:** Gradio's dark default theme masks native browser focus outlines, leaving keyboard users without visible focus indication on interactive controls like textareas, buttons, and inputs.
**Action:** Always inject high-contrast `:focus-visible` CSS rules scoped with `.gradio-container` and `!important` flags for keyboard focus accessibility.
