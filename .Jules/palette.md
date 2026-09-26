## 2026-03-30 - Focus Visible Styles for Gradio Dark Cyberpunk Theme
**Learning:** Gradio UI components override standard element focus indicators, making keyboard navigation invisible against dark/cyberpunk themes. Custom `:focus-visible` styles with elevated specificity (`.gradio-container`) and `!important` outline/box-shadow rules are necessary to maintain web accessibility standards.
**Action:** Always apply explicit `.gradio-container *:focus-visible` ring/outline rules when customizing Gradio dark themes.
