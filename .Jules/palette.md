## 2026-03-30 - High-contrast focus indicators in Gradio dark themes
**Learning:** Gradio's internal CSS components override browser defaults and often suppress outline focus indicators on dark themes, making keyboard navigation difficult for screen reader and keyboard-only users.
**Action:** Always declare explicit `.gradio-container button:focus-visible, .gradio-container input:focus-visible, .gradio-container textarea:focus-visible, .gradio-container a:focus-visible` CSS rules using high-contrast outlines and offsets (`!important`) matching the primary theme palette.
