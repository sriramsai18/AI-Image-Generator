## 2026-03-30 - Gradio High-Contrast Focus-Visible Outline Overrides
**Learning:** Gradio's default theme styles do not provide sufficient outline contrast in dark mode, and internal components may suppress outlines. Adding explicit `:focus-visible` rules targeted with `.gradio-container` and `!important` ensures standard keyboard focus outlines are clear and accessible.
**Action:** Always include high-contrast `:focus-visible` rules with `outline` and `box-shadow` properties when styling custom Gradio dark themes.
