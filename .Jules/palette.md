## 2024-10-26 - Keyboard Focus Indicators in Custom Gradio Dark Themes
**Learning:** Gradio's internal UI components often inject CSS styles with `!important` flags that strip default focus outlines, making keyboard navigation invisible on dark cyberpunk themes.
**Action:** Always target interactive elements within `.gradio-container` using higher specificity `:focus-visible` rules with explicit `outline` and `box-shadow` properties to ensure keyboard accessibility.
