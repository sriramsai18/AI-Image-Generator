## 2025-09-04 - High-Contrast Keyboard Focus Indicators in Gradio Dark Themes
**Learning:** Gradio dark themes often override standard input border/outline focus styles, making keyboard navigation difficult or invisible for screen reader and keyboard-only users.
**Action:** Always inject high-specificity `.gradio-container *:focus-visible` rules with bright outline colors (`#39ff14`) and `!important` flags so keyboard focus states are clearly visible across inputs, textareas, buttons, and links.
