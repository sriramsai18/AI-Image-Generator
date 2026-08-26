## 2024-08-26 - High-Contrast Keyboard Focus Indicators for Gradio Dark Themes
**Learning:** Gradio's internal UI components apply `outline: none !important;` on input fields and buttons, completely suppressing default browser focus rings. In dark/cyberpunk themes, keyboard users lose visual indication of where focus is located.
**Action:** Always add explicit `.gradio-container *:focus-visible` styling with `!important` on `outline` and `box-shadow` matching the theme highlight color to guarantee keyboard navigation accessibility.
