## 2026-10-28 - Focus Indicators on Custom Dark Themes in Gradio
**Learning:** Gradio components internal CSS often sets `outline: none !important;` on form inputs and buttons, which hides focus indicators during keyboard navigation on dark background interfaces.
**Action:** Always add high-contrast `:focus-visible` CSS rules prefixed with `.gradio-container` and using `!important` on `outline` and `box-shadow` properties to ensure focus visibility.
