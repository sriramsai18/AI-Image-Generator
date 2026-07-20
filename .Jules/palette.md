# Palette's Journal - Critical UX/Accessibility Learnings

## 2026-07-20 - Custom Gradio Cyberpunk Stylesheet Focus Indicators
**Learning:** When developing highly customized dark/cyberpunk stylesheets inside Gradio, standard interactive elements (like buttons, range inputs, textareas, and links) can have their default browser outlines and focus indicators completely hidden or blending into dark backgrounds. This severely limits keyboard accessibility (A11y) and makes Tab navigation unusable. Explicit high-contrast `:focus-visible` CSS rules are required to ensure safe, accessible navigation without impacting normal click-states.
**Action:** Always include high-contrast `:focus-visible` outline styles with an appropriate outline-offset and neon box-shadow in custom styles of Gradio blocks to maintain compliance with keyboard navigation standards.
