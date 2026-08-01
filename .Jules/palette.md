# Palette's Journal

🎨 A diary of micro-UX and accessibility improvements.

## 2025-08-01 - Overriding Gradio Custom Theme Focus Indicators
**Learning:** Gradio’s internally compiled React/Svelte components inject heavy inline styles or rules containing `!important` (such as `outline: none !important;` or default grey borders) on interactive elements like `textarea`, `input`, and `button`. Standard tag-level CSS selectors (e.g., `textarea:focus-visible`) fail to override these default focus borders. To solve this, custom stylesheets must use selectors with higher specificity by wrapping them inside the `.gradio-container` top-level wrapper class, and appending `!important` to both the `outline` and `box-shadow` styles.
**Action:** When designing dark or cyberpunk neon themes in Gradio, always target `:focus-visible` states explicitly with `.gradio-container` prefix class selectors to ensure keyboard accessibility focus rings are distinct and meet WCAG guidelines.
