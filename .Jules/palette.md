## 2025-10-02 - High-contrast keyboard focus indicators in Gradio dark themes
**Learning:** Gradio 3.x default dark theme stylesheet lacks explicit focus indicators (`:focus-visible`) for buttons, textareas, and interactive components, leaving keyboard users with low or invisible focus states during Tab navigation.
**Action:** Explicitly define high-contrast `:focus-visible` CSS rules (`outline: 2px solid <theme-color> !important; outline-offset: 2px !important;`) for buttons, inputs, textareas, and links in custom Gradio stylesheets.
