## 2025-09-13 - High-Contrast Focus Indicators in Custom Gradio Themes
**Learning:** Gradio default styles often lack noticeable keyboard focus outlines, especially when custom dark backgrounds are applied. Using explicit `.gradio-container *:focus-visible` rules with `!important` ensures keyboard navigation is clearly visible without impacting mouse users.
**Action:** Always add high-contrast `:focus-visible` outlines and offsets when styling custom Gradio apps.
