# Palette's Journal - AI Image Generator UX & Accessibility

## 2025-02-17 - Keyboard Focus in Dark/Neon Themes
**Learning:** High-contrast keyboard focus indicators (`:focus-visible`) are often omitted or broken in dark cyberpunk designs because browser defaults clash with the theme. Using a theme-consistent neon green (`#39ff14`) matching the UI's existing status text maintains visual cohesion while providing superior high-contrast outline feedback for accessibility.
**Action:** Always map keyboard outline styles to active/high-contrast neon tokens in dark theme applications to satisfy WCAG focus visibility standards elegantly.

## 2025-02-17 - Advanced Inputs Reset Loop
**Learning:** Complex AI generation tools with many numerical sliders (CFG, Steps, Width, Height) often discourage experimentation because users fear "losing" their good settings or have to perform tedious manual resets. A clean secondary "Reset Defaults" micro-UX allows friction-free exploration and recovery.
**Action:** Add a secondary reset button next to primary actions whenever complex generation parameters are exposed to the user.
