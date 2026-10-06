## 2024-10-06 - High Contrast Focus Indicators in Custom Dark Themes
**Learning:** Gradio custom dark/cyberpunk stylesheets can obscure default focus outlines, making keyboard navigation difficult or invisible for screen readers and keyboard users.
**Action:** Always declare explicit, high-contrast `:focus-visible` outline styles (`outline: 2px solid <color> !important; outline-offset: 2px !important;`) on `.gradio-container` interactive elements (`button`, `input`, `textarea`, `a`, `[role="button"]`) when customizing theme CSS.
