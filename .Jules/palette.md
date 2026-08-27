## 2026-02-27 - Gradio Dark Theme Keyboard Focus Contrast
**Learning:** Gradio's dark container styling removes standard default focus outlines for interactive elements, making keyboard navigation (`Tab` key) visually invisible.
**Action:** Always inject high-contrast `:focus-visible` CSS rules for buttons, inputs, textareas, and links targeting `.gradio-container` with `!important` flags.
