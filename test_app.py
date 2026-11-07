import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock diffusers before importing app
mock_diffusers = MagicMock()
mock_pipe = MagicMock()
mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipe
mock_pipe.to.return_value = mock_pipe
sys.modules["diffusers"] = mock_diffusers

import gradio.networking

# Mock url_ok to avoid localhost probe failures in sandboxed environment
gradio.networking.url_ok = lambda url: True

from playwright.sync_api import sync_playwright

import app


class TestAppUIAndUX(unittest.TestCase):
    def test_app_loaded(self):
        self.assertIsNotNone(app.demo)

    def test_css_focus_visible_styles(self):
        # Verify :focus-visible accessibility styles are defined in custom css
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #e63946 !important", app.css)
        self.assertIn("button.secondary", app.css)

    def test_reset_button_in_blocks(self):
        # Verify reset button is defined in Gradio Blocks components
        button_labels = [
            comp.value for comp in app.demo.blocks.values()
            if hasattr(comp, "value") and isinstance(comp.value, str)
        ]
        self.assertIn("↺ RESET", button_labels)
        self.assertIn("▶ GENERATE IMAGE", button_labels)

    def test_generate_image_validation(self):
        # Empty prompt returns validation error
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    @patch("app.pipe")
    def test_generate_image_mock_success(self, mock_pipeline):
        mock_result = MagicMock()
        mock_img = MagicMock()
        mock_result.images = [mock_img]
        mock_pipeline.return_value = mock_result

        img, info = app.generate_image("a cute cat", "", 20, 7.0, 512, 512, 42)
        self.assertEqual(img, mock_img)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

    def test_frontend_interaction_and_reset(self):
        # Launch demo on a local port with prevent_thread_lock=True
        port = 7865
        app.demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            show_error=True,
        )

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")

                # Check title / header
                header = page.locator(".app-title")
                self.assertTrue(header.is_visible())

                # Check Reset and Generate buttons exist
                reset_btn = page.locator("button", has_text="↺ RESET")
                generate_btn = page.locator("button", has_text="▶ GENERATE IMAGE")
                self.assertTrue(reset_btn.is_visible())
                self.assertTrue(generate_btn.is_visible())

                # Fill in prompt text area
                prompt_textarea = page.locator("textarea").first
                sample_prompt = "A neon cyberpunk city with flying cars"
                prompt_textarea.fill(sample_prompt)
                self.assertEqual(prompt_textarea.input_value(), sample_prompt)

                # Test keyboard navigation / focus visible
                prompt_textarea.focus()
                page.keyboard.press("Tab")

                # Click Reset button and verify prompt gets cleared
                reset_btn.click()
                page.wait_for_timeout(300)
                self.assertEqual(prompt_textarea.input_value(), "")
                browser.close()
        finally:
            app.demo.close()


if __name__ == "__main__":
    unittest.main()
