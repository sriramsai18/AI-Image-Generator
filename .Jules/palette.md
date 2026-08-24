## 2026-08-24 - High-Contrast Keyboard Focus Indicators in Dark Gradio Themes
**Learning:** Default Gradio focus rings can lose contrast or be overridden on custom dark or cyberpunk themes. Adding specific `.gradio-container :focus-visible` rules with `!important` ensures standard keyboard tab navigation remains highly visible and accessible without breaking hover or click aesthetics.
**Action:** Always include high-contrast `:focus-visible` CSS rules scoped to `.gradio-container` for buttons, inputs, textareas, and links on custom dark themes.
