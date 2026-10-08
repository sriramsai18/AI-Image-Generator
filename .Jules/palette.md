## 2025-05-10 - Keyboard Focus Indicators in Custom Gradio Dark Themes
**Learning:** Gradio's default theme styles do not provide high-contrast focus indicators on custom dark backgrounds, leaving keyboard users without visual focus feedback. Standard tag-level CSS selectors (`button:focus`) are easily overridden by internal Gradio CSS rules (`outline: none !important`).
**Action:** Scope `:focus-visible` accessibility selectors with `.gradio-container` and specify high-contrast `outline` and `box-shadow` with `!important` to ensure accessible keyboard navigation.
