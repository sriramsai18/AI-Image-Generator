## 2025-02-21 - Gradio Focus Outline Override Specificity
**Learning:** Gradio's internal components inject default CSS rules with `!important` (such as `outline: none !important;` on inputs and textareas) that override standard tag-level focus rules.
**Action:** Always scope custom keyboard focus accessibility styles using higher specificity selectors (e.g. prefixing with `.gradio-container`) and apply `!important` on `outline` and `box-shadow` properties to ensure high-contrast `:focus-visible` indicators are rendered correctly.
