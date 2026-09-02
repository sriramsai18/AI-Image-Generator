## 2026-09-02 - High-Contrast Keyboard Focus Indicators for Custom Dark Themes
**Learning:** Custom dark cyberpunk stylesheets in Gradio override default browser focus states, making interactive elements invisible or difficult to distinguish for keyboard users without explicit `:focus-visible` rules.
**Action:** Always define explicit `:focus-visible` CSS rules with prominent high-contrast outlines and neon box-shadows on interactive controls (`button`, `input`, `textarea`, `a`, `select`).
