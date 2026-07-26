## 2025-07-26 - [Initialization]
**Learning:** In Python 3.12+ environments, pinning `torch==2.1.0` fails to install due to pip distribution matching rules. Using `torch>=2.1.0` solves this. Gradio 3.50.2 needs `starlette<0.28.0` and `fastapi<0.100.0` to avoid Starlette template context crashes.
**Action:** Use `torch>=2.1.0`, `starlette<0.28.0`, and `fastapi<0.100.0` in Python 3.12 projects running Gradio 3.50.2.
