## 2024-05-20 - High-Contrast Keyboard Focus Indicators in Gradio
**Learning:** Gradio components override standard focus rings with low-contrast or hidden defaults (`outline: none !important;`). To ensure visible keyboard focus, custom stylesheets must target interactive tags (`button`, `input`, `textarea`, `a`, `[role="button"]`) with `:focus-visible` scoped under `.gradio-container` using `!important`.
**Action:** Always include high-contrast `:focus-visible` styling with `!important` on `.gradio-container` when customizing Gradio dark or cyberpunk themes.
