## 2026-03-08 - High-Contrast Keyboard Focus Indicators
**Learning:** Gradio's default theme in dark mode lacks prominent keyboard focus indicators on interactive elements (`button`, `input`, `textarea`, `a`), making keyboard navigation difficult for accessibility.
**Action:** Always declare explicit, high-contrast `:focus-visible` styles with `outline` and `box-shadow` on container interactive elements in custom Gradio CSS.
