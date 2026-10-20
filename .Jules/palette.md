## 2024-05-20 - High-Contrast Keyboard Focus Indicators in Gradio
**Learning:** Gradio's default input and button focus styles are easily overridden by dark theme custom CSS, rendering keyboard focus invisible.
**Action:** Always add explicit `.gradio-container *:focus-visible` CSS declarations with `!important` on `outline` and `box-shadow` when customizing Gradio dark themes.
