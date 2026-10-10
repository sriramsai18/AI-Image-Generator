## 2026-10-10 - High-Contrast Keyboard Focus Indicators in Dark Gradio Themes
**Learning:** Custom dark/neon themes in Gradio often override input and button styles without explicitly specifying `:focus-visible` focus rings. Combined with Gradio's default `outline: none !important;`, keyboard focus navigation becomes invisible for non-mouse users.
**Action:** Always inject high-contrast `:focus-visible` CSS rules on interactive tags (`button`, `input`, `textarea`, `a`, `summary`, `[role="button"]`) with higher specificity and `!important` outline/box-shadow properties.
