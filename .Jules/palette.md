## 2026-03-18 - High-Contrast Keyboard Focus Indicators in Dark Cyberpunk UI
**Learning:** Custom dark/neon stylesheets for Gradio containers can obscure default browser focus outlines, making keyboard navigation difficult for screen reader and keyboard-only users.
**Action:** Explicitly define high-contrast `:focus-visible` styling (`outline: 2px solid #e63946 !important; outline-offset: 2px !important;`) on interactive elements (`button`, `input`, `textarea`, `a`) to ensure accessible keyboard navigation.
