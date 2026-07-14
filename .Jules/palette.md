## 2025-10-24 - [Accessible custom dark theme focus indicator styles & quick-reset UX features]
**Learning:** Custom styled dark and neon web pages often completely hide or override browser-default keyboard focus rings, making them highly inaccessible for assistive tech or keyboard-only navigators. Applying standard high-contrast `:focus-visible` rules ensures focus status is clearly visible without compromising standard mouse interaction visual styles.
**Action:** Always include high-contrast focus rings with `:focus-visible` on custom-themed Gradio and web forms so interactive elements remain clear to keyboard users.

## 2025-10-24 - [Gradio 3.50.2 Starlette and FastAPI Compatibility]
**Learning:** Gradio 3.50.2 requires Starlette < 0.28.0 and FastAPI < 0.100.0 to prevent template rendering crashes (`TypeError: unhashable type: 'dict'`). Installing newer versions of FastAPI or Starlette on Python 3.12 with Gradio 3.x results in unhandled dictionary hash errors on load.
**Action:** Explicitly restrict FastAPI and Starlette to lower matching versions when running Gradio 3.50.2.
