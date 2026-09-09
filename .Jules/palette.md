## 2026-03-31 - High-Contrast Keyboard Focus Indicators
**Learning:** Gradio's default input styling suppresses outlines, which makes keyboard navigation invisible for users who rely on Tab navigation. High-contrast `:focus-visible` styles with explicit `outline` and `box-shadow` ensure keyboard accessibility across custom dark theme components.
**Action:** Always include high-contrast `:focus-visible` rules with `!important` when overriding Gradio CSS in dark-themed interfaces.
