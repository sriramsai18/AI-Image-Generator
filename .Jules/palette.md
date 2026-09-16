## 2024-09-16 - High-Contrast Keyboard Focus States in Dark Theme
**Learning:** Gradio dark themes often override or obscure default browser focus rings on input fields and buttons, reducing keyboard accessibility for users relying on Tab navigation.
**Action:** Always declare explicit, high-contrast `:focus-visible` styles scoped under `.gradio-container` with `!important` on `outline` and `box-shadow` for interactive elements (`button`, `input`, `textarea`, `a`).
