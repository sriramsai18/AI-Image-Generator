# Palette's Journal - Critical UX/A11y Learnings

## 2025-07-16 - Cyberpunk Theme Keyboard Focus Visibility Deficiencies
**Learning:** In applications utilizing custom dark backgrounds (e.g. `#080b0f` and `#0f1318`) alongside custom neon styling, the standard browser focus indicators are virtually invisible or completely aesthetic-breaking. This renders the interface entirely inaccessible to keyboard-only users who cannot track their active focus ring as they navigate.
**Action:** Always declare explicit, high-contrast, theme-consistent `:focus-visible` outlines and glowing box-shadow rules targeting all interactive tags (`button`, `input`, `textarea`, `a`, `input[type="range"]`, and any custom component classes like Gradio's `.gr-button`) to guarantee accessible and beautiful keyboard navigation.
