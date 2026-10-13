## 2024-05-20 - High-Contrast Keyboard Focus Indicators in Gradio
**Learning:** Gradio's default input styling suppresses focus outlines with `outline: none !important`, making keyboard navigation invisible on dark cyberpunk themes.
**Action:** Always declare explicit `.gradio-container *:focus-visible` styles with `!important` on `outline` and `box-shadow` for standard interactive elements.
