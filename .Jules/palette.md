## 2025-05-15 - High-Contrast Focus-Visible Styles for Dark Cyberpunk Theme
**Learning:** Gradio's default input focus styles blend into dark custom themes, leaving keyboard users without clear visual feedback on focused elements.
**Action:** Always declare explicit `:focus-visible` CSS rules prefixed with `.gradio-container` and `!important` on `outline` and `box-shadow` for interactive elements (`button`, `input`, `textarea`, `a`) in custom Gradio themes.
