## 2026-02-28 - High-Contrast Focus Indicators in Gradio Dark Themes
**Learning:** Gradio's internal UI components inject `outline: none !important` by default, suppressing focus rings on inputs, buttons, links, and accordion elements when dark themes are applied.
**Action:** Ensure custom dark stylesheets declare `.gradio-container <tag>:focus-visible` with high-contrast outlines and `!important` to preserve keyboard navigation accessibility.
