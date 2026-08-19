## 2026-08-19 - Focus Indicators in Custom Dark Themes
**Learning:** Default Gradio component styles in dark mode lack clear keyboard focus indicators. Standard CSS `:focus` or `:focus-visible` selectors on bare tags (`button`, `input`, `textarea`) can be overridden or hidden by Gradio's default theme styles.
**Action:** Always scope custom `:focus-visible` styles under `.gradio-container` with higher specificity and `!important` to ensure accessible, high-contrast focus rings for keyboard users across custom dark/neon themes.
