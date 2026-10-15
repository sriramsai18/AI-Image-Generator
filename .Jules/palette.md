# Palette's Journal

## 2026-03-31 - High-contrast focus indicators in Gradio dark themes
**Learning:** Gradio 3.x internal components set `outline: none !important` on input/textarea focus states, making keyboard navigation focus invisible in custom dark themes.
**Action:** Use higher specificity selectors prefixed with `.gradio-container` and `!important` on `outline` and `box-shadow` properties to ensure focus indicators remain visible for keyboard accessibility.
