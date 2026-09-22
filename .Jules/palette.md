## 2024-09-22 - High-contrast :focus-visible in Gradio Dark Themes
**Learning:** Gradio's internal UI components apply default outline resets (`outline: none !important`) that obscure standard browser keyboard focus indicators, making dark-themed apps virtually unnavigable for keyboard users.
**Action:** Always declare explicit, high-specificity `:focus-visible` styles (`.gradio-container button:focus-visible`, etc.) with `outline` and `box-shadow` to guarantee keyboard accessibility in dark mode interfaces.
