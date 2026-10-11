## 2024-10-11 - High Contrast Focus Indicators in Custom Dark Themes
**Learning:** Gradio default input focus styles can blend into dark custom backgrounds or be overridden by internal component styling, making keyboard navigation difficult for accessibility users.
**Action:** Always declare explicit `.gradio-container *:focus-visible` high-contrast outline rules with `!important` to guarantee keyboard navigation focus rings remain clear against custom dark backgrounds.
