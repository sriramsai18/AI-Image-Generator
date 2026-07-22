# Palette's Journal

## 2025-06-21 - [High-Contrast Keyboard Focus Indicators in Dark/Neon Theme]
**Learning:** In highly customized dark/neon (cyberpunk) styled Gradio applications, default browser focus rings (which are typically light-blue or subtle gray) are practically invisible. This prevents keyboard-only and screen-reader users from visually tracking their active selection, creating a severe accessibility block. Declaring explicit `:focus-visible` styles with a neon accent color ensures the UI is accessible while keeping the aesthetic consistent.
**Action:** Always provide explicit, high-contrast `:focus-visible` styles (using outline or box-shadow with the theme's highlight color) for all standard interactive tags (button, input, textarea, a, slider) when designing custom dark or high-contrast styles.
