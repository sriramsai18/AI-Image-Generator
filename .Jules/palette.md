## 2026-02-27 - High-contrast keyboard focus indicators in Gradio dark theme
**Learning:** Gradio components override default browser focus styles in custom dark themes, making keyboard navigation invisible or hard to track.
**Action:** Use `.gradio-container button:focus-visible, .gradio-container input:focus-visible, .gradio-container textarea:focus-visible, .gradio-container a:focus-visible` with `!important` on outline and box-shadow to guarantee keyboard focus feedback.
