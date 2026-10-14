## 2025-10-14 - High-Contrast Focus Visible States in Custom Dark Gradio Themes
**Learning:** Default browser focus rings are suppressed or visually obscured when using dark custom CSS themes in Gradio, reducing accessibility for keyboard users navigating inputs and buttons.
**Action:** Explicitly define high-contrast `:focus-visible` styles scoped to `.gradio-container` with `!important` declarations on `outline` and `box-shadow`.
