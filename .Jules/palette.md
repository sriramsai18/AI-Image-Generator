## 2024-09-29 - Gradio Focus Indicator Specificity Override
**Learning:** Gradio's internal CSS injects `outline: none !important;` on form controls and buttons, obscuring default browser focus rings for keyboard navigation.
**Action:** When defining custom stylesheets in Gradio, always use higher-specificity selectors prefixed with `.gradio-container` and apply `!important` on `:focus-visible` properties (`outline` and `box-shadow`) to guarantee keyboard accessibility.
