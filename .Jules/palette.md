# Palette's Journal - Critical UX Learnings

## 2026-07-17 - [Custom Dark Theme Focus Indicators & Input Resets]
**Learning:** In heavily stylized custom dark/neon interfaces (like cyberpunk or futuristic terminal styles), standard browser focus indicators completely blend in or disappear against dark background colors (#080b0f). This entirely breaks keyboard navigation accessibility (a11y). Explicit high-contrast `:focus-visible` styles matching the brand color (#e63946) with a glow/shadow offset maintain aesthetic consistency while restoring robust focus tracking.
Furthermore, in multi-field GenAI configurations (combining prompts, negative prompts, and advanced parameters like CFG, steps, seed), user input friction builds up quickly. A primary action button (Generate) should be balanced with a clear secondary utility (Reset) to reduce manual form clearing, especially when parameters are tucked away under collapsible accordions.

**Action:** Always verify keyboard accessibility on any custom-styled stylesheet. When styling a primary action, check if form elements can benefit from a secondary theme-aligned "Reset" or "Clear" interaction under a simple button.
