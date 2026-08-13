# Palette UX & Accessibility Journal

## 2024-08-13 - Gradio Focus Indicator Overrides
**Learning:** Gradio’s internal web components inject custom CSS rules with `!important` tags that aggressively suppress standard interactive element keyboard focus styling (such as `outline: none !important;`). To successfully override these defaults for keyboard navigability and ensure consistent custom cyberpunk theme-appropriate `:focus-visible` states, one must declare a highly specific selector chain (e.g. `.gradio-container *:focus-visible`) coupled with `!important` declarations on both outline and box-shadow properties.
**Action:** Always inspect the target environment's CSS compiled style output when custom themes are in play. Apply high specificity rules targeting `:focus-visible` using `.gradio-container` prefix wrappers to safeguard key accessibility markers against style erasure.
