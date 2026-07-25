# Palette's Journal 🎨

## 2025-07-25 - High-Contrast Focus Indicators in Dark/Cyberpunk Themes
**Learning:** Default browser focus rings often become nearly invisible in dark or neon/cyberpunk CSS themes because their default styles have extremely low contrast against deep black/gray backgrounds. To prevent keyboard-navigating users from losing their cursor position, explicit high-contrast `:focus-visible` outline rules should be declared. Utilizing theme-matching but highly vibrant colors (such as terminal neon green `#39ff14` on dark gray backgrounds) preserves the aesthetic while providing clear visual guidance.
**Action:** Always inspect the visual focus state of all interactive elements during keyboard navigation (tabbing) in dark/custom themes. Proactively add explicit `:focus-visible` styles with sufficient outline contrast and custom outline-offsets.

## 2025-07-25 - Secondary Action Resets
**Learning:** Complex input generation tools (like AI Image Generators with multiple prompt boxes, sliders, and seed parameters) can suffer from cognitive load and user fatigue when users want to clear or restart their configuration. A low-profile, secondary action button like "↺ RESET" next to the primary CTA helps users easily return to a clean/default slate in one click without cluttering the main UI.
**Action:** For forms or generators with more than 3-4 interactive sliders and text inputs, always include an easily-discoverable but styled-secondary reset/clear option.
