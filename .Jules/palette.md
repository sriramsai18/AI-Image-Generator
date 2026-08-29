## 2024-08-29 - High-Contrast Focus-Visible Indicators for Custom Dark Themes in Gradio
**Learning:** Gradio's default input styling can obscure standard focus indicators, making keyboard navigation invisible against dark/cyberpunk background themes.
**Action:** Always inject high-contrast `:focus-visible` styles with `outline` and `box-shadow` scoped under `.gradio-container` for interactive elements (`button`, `input`, `textarea`, `a`).
