## 2026-10-23 - Focus Visible Indicators in Gradio Dark Themes
**Learning:** Gradio default styles override native browser outline focus indicators on inputs, textareas, and buttons, making keyboard navigation difficult in custom dark themes.
**Action:** Always add high-contrast `:focus-visible` rules scoped with `.gradio-container` and `!important` to maintain visual clarity for keyboard users.
