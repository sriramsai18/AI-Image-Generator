## 2026-09-30 - Keyboard Focus-Visible Styling in Gradio Dark Themes
**Learning:** Gradio's dark theme defaults obscure focused inputs and buttons when navigating via keyboard (Tab/Shift+Tab). Adding explicit `.gradio-container :focus-visible` rules with a high-contrast outline (`#e63946`) and neon glow ensures visual accessibility without affecting mouse click states.
**Action:** Always include high-contrast `:focus-visible` CSS rules targetting inputs, textareas, buttons, and links when designing dark-mode Gradio interfaces.
