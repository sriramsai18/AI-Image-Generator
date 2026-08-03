# Palette's Journal

## 2024-08-03 - High-Contrast Keyboard Focus Styling in Gradio Apps
**Learning:** In highly customized dark or cyberpunk themes in Gradio, standard browser focus outlines are completely invisible or suppressed. Since Gradio has complex, highly specific internal CSS rules (often using `!important` on elements like inputs and textareas), default tag-level `:focus-visible` styles fail to apply. High specificity selectors prefixed with `.gradio-container` combined with `!important` on the outline and box-shadow properties are necessary to override Gradio's defaults and ensure keyboard accessibility.
**Action:** Always use high-specificity selectors (such as `.gradio-container button:focus-visible`) with `!important` on `outline` and `box-shadow` properties to ensure focus states stand out in heavily customized Gradio interfaces.
