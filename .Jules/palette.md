## 2024-10-19 - Keyboard Focus Visible Indicators in Gradio Dark Themes
**Learning:** Custom dark mode CSS in Gradio apps often overrides default focus outlines (`textarea:focus, input:focus`), causing keyboard users to lose track of focus during Tab navigation.
**Action:** Always include high-contrast `:focus-visible` CSS rules for standard interactive elements (`button`, `textarea`, `input`, `a`) that match the application's brand accent color.
