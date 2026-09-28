## 2026-09-28 - High-contrast focus-visible indicators in Gradio dark themes
**Learning:** Gradio default styles inject `outline: none !important` on textareas and inputs, hiding standard browser focus rings on dark themes. Adding high-contrast `:focus-visible` styles scoped under `.gradio-container` restores visual focus indicators for keyboard navigation while avoiding persistent focus rings on click.
**Action:** Always declare `.gradio-container *:focus-visible` rules with `!important` on `outline` and `box-shadow` when customizing dark Gradio user interfaces.
