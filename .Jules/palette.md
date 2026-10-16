## 2025-02-14 - High-Contrast Focus Visible States in Custom Dark Gradio Themes
**Learning:** Gradio component inputs and buttons inject default CSS rules that can conceal standard browser focus outlines in dark cyberpunk themes.
**Action:** Always declare explicit, high-contrast `:focus-visible` styles with `!important` on `.gradio-container` interactive elements (`button`, `input`, `textarea`, `a`, `summary`) to ensure keyboard accessibility navigation is clearly visible.
