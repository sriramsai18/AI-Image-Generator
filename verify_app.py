import os
import time

import diffusers

# 1. Mocking StableDiffusionPipeline and torch BEFORE importing app
import torch


# Create a mock pipeline class
class MockUnet:
    def __init__(self):
        pass
    def to(self, *args, **kwargs):
        return self

class MockVae:
    def __init__(self):
        pass
    def to(self, *args, **kwargs):
        return self

class MockImages:
    def __init__(self):
        # Return a dummy solid color image (e.g., green square)
        from PIL import Image
        self.images = [Image.new("RGB", (512, 512), color="green")]

class MockPipeline:
    def __init__(self):
        self.unet = MockUnet()
        self.vae = MockVae()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        print("[MOCK] Pipeline.from_pretrained called")
        return cls()

    def to(self, *args, **kwargs):
        print(f"[MOCK] Moving pipeline to {args} {kwargs}")
        return self

    def enable_attention_slicing(self):
        print("[MOCK] Attention slicing enabled")

    def __call__(self, *args, **kwargs):
        print(f"[MOCK] Pipeline call with prompt: {kwargs.get('prompt')}")
        return MockImages()

# Apply the mocks
diffusers.StableDiffusionPipeline.from_pretrained = MockPipeline.from_pretrained
torch.cuda.is_available = lambda: False

# Now import the app (which will load our MockPipeline)
# Playwright imports
from playwright.sync_api import sync_playwright

import app


def run_cuj():
    # Launch Gradio server
    print("Launching mocked Gradio server...")
    server_port = 7865
    app.demo.launch(server_port=server_port, prevent_thread_lock=True)

    time.sleep(2)  # Wait for server to start

    print("Starting Playwright verification...")
    # Create output directories
    os.makedirs("/home/jules/verification/videos", exist_ok=True)
    os.makedirs("/home/jules/verification/screenshots", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/home/jules/verification/videos"
        )
        page = context.new_page()
        try:
            # Go to local app
            url = f"http://127.0.0.1:{server_port}"
            print(f"Navigating to {url}")
            page.goto(url)
            page.wait_for_timeout(1000)

            # Fill in the prompt
            print("Filling prompt...")
            prompt_textarea = page.get_by_label("PROMPT", exact=True)
            prompt_textarea.fill("A neon cyberpunk cat wearing sunglasses, highly detailed, 4k")
            page.wait_for_timeout(1000)

            # Click generate
            print("Clicking GENERATE IMAGE button...")
            generate_btn = page.get_by_role("button", name="▶ GENERATE IMAGE")
            generate_btn.click()

            # Wait for generation to complete (calls mock)
            print("Waiting for generation...")
            page.wait_for_timeout(3000)

            # Take screenshot
            screenshot_path = "/home/jules/verification/screenshots/verification.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot taken at {screenshot_path}")

        finally:
            context.close()
            browser.close()
            # Stop gradio server
            app.demo.close()
            print("Gradio server closed.")

if __name__ == "__main__":
    run_cuj()
