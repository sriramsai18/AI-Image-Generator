## 2025-02-18 - High-contrast focus-visible rings in dark Gradio themes
**Learning:** Custom Gradio dark stylesheets can hide or suppress browser default focus indicators, reducing keyboard navigation clarity.
**Action:** Always include high-contrast `:focus-visible` CSS rules on interactive elements (`button`, `a`, `input`, `textarea`) scoped under `.gradio-container` with `!important` flags.
