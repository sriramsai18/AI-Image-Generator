# Palette UX Learnings

## Keyboard Accessibility Focus Indicators
**Learning:** Default browser outline focus indicators often blend in or disappear completely when using custom dark/neon stylesheets. Declaring explicit high-contrast `:focus-visible` styles on interactive tags (button, input, textarea, a) guarantees that keyboard-only users can navigate the interface effectively.
**Action:** Add high-contrast `:focus-visible` outline styles with an appropriate offset in all custom Gradio stylesheets.
