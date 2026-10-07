# Palette's Journal

## 2025-05-20 - High-Contrast Keyboard Focus in Gradio Custom Themes
**Learning:** Custom Gradio CSS themes with dark/cyberpunk backgrounds often obscure default focus rings on interactive elements (buttons, inputs, links, textareas) unless explicit high-contrast `:focus-visible` rules with `!important` and higher specificity are defined on `.gradio-container`.
**Action:** Always include high-contrast `:focus-visible` CSS rules for standard interactive tags when customizing Gradio UI components.
