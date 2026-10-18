## 2026-03-31 - Gradio Keyboard Focus Indicators
**Learning:** Gradio's internal components often inject CSS rules with `!important` (such as `outline: none !important;`) on input elements, hiding default browser focus indicators during keyboard navigation.
**Action:** Use higher specificity selectors prefixed with `.gradio-container` and `:focus-visible` with `!important` on `outline` and `box-shadow` properties to ensure high-contrast focus visibility.
