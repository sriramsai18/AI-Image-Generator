# Palette's Journal - Critical UX/a11y Learnings Only

## 2025-02-14 - Custom CSS Theme Focus States Accessibity
**Learning:** When applying custom dark/cyberpunk themes in Gradio, standard interactive elements' default browser focus indicators can blend in or completely disappear, posing a severe keyboard accessibility blocker. Explicit high-contrast `:focus-visible` styles should be added to basic tag selectors like buttons, inputs, textareas, anchors, and elements with `role="button"`.
**Action:** Always include a dedicated `:focus-visible` CSS selector rule block targeting standard interactive elements to guarantee clear visual focus indication.
