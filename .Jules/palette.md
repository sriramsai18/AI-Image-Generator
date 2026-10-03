## 2025-02-23 - Focus Visible Indicators in Gradio Dark Themes
**Learning:** Gradio inputs and buttons override browser default focus rings, which can render keyboard focus invisible on dark background themes.
**Action:** Always declare explicit `:focus-visible` CSS rules scoped with `.gradio-container` with custom outline and box-shadow rings matching the theme's high-contrast accent colors.
