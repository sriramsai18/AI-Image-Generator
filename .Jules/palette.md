# Palette's Journal

Critical UX and accessibility learnings only.

## 2026-05-19 - High-Contrast Keyboard Focus Rings in Dark Cyberpunk UIs
**Learning:** In custom dark-mode Gradio apps, default browser focus rings (e.g. `rgb(16, 16, 16)` or dark outlines) become completely invisible against dark `#080b0f` backgrounds, and Gradio's internal CSS injects `outline: none !important;` on textareas and inputs. High-contrast `:focus-visible` outlines must be scoped under `.gradio-container` with `!important` to restore WCAG-compliant keyboard navigation without altering mouse-click aesthetics.
**Action:** Always inspect computed styles for `:focus-visible` across buttons, inputs, links, and accordion headers when customizing Gradio themes, applying high-contrast outlines (2px solid #e63946 with offset and glow) using high-specificity selectors.
