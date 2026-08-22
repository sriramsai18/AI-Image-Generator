## 2026-08-22 - Gradio High-Contrast Focus Visible Styling
**Learning:** Gradio 3.x default UI components often lack explicit, visible `:focus-visible` styling or rely on low-contrast focus outlines, making keyboard navigation difficult.
**Action:** Always add explicit, high-contrast `:focus-visible` CSS rules for interactive elements (`button`, `input`, `textarea`, `a`) scoped with higher specificity under `.gradio-container`.
