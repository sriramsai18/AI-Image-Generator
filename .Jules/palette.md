## 2026-09-03 - High-Contrast Keyboard Focus in Gradio Dark Themes
**Learning:** Gradio default component styles override tag-level focus rules with `outline: none !important`, making dark cyberpunk themes inaccessible to keyboard users. Using higher specificity selectors (`.gradio-container *:focus-visible`) with explicit `outline`, `outline-offset`, and neon `box-shadow` ensures accessible focus visibility.
**Action:** Always scope `:focus-visible` rules under `.gradio-container` with high-contrast styles and `!important` flags when customizing Gradio application themes.
