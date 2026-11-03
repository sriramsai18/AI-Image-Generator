import os
import unittest
from unittest.mock import patch

os.environ["MOCK_MODE"] = "1"
import app


class TestApp(unittest.TestCase):
    def test_mock_pipeline_returns_image(self):
        pipe = app.MockPipeline()
        result = pipe("a futuristic cyberpunk city", width=256, height=256)
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.images[0].size, (256, 256))

    def test_generate_image_empty_prompt(self):
        img, info = app.generate_image("", "ugly", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("⚠️ Please enter a prompt first!", info)

    def test_generate_image_valid_prompt(self):
        img, info = app.generate_image(
            "a lone tree on a hill", "blurry", 20, 7.0, 512, 512, 42
        )
        self.assertIsNotNone(img)
        self.assertIn("✅ Generated in", info)
        self.assertIn("Steps: 20", info)
        self.assertIn("CFG: 7.0", info)
        self.assertIn("Seed: 42", info)

    def test_generate_image_random_seed(self):
        img, info = app.generate_image(
            "a lone tree on a hill", "blurry", 15, 6.5, 512, 512, -1
        )
        self.assertIsNotNone(img)
        self.assertIn("Seed: random", info)

    def test_focus_visible_css_present(self):
        self.assertIn(":focus-visible", app.css)
        self.assertIn(".gradio-container button:focus-visible", app.css)
        self.assertIn("#00f0ff", app.css)

    @patch("gradio.networking.url_ok", return_value=True)
    def test_demo_blocks_instance(self, mock_url_ok):
        self.assertIsNotNone(app.demo)
        self.assertIn("Text2Image", app.demo.title)

    def test_pipe_exception_handling(self):
        with patch.object(app, "pipe", side_effect=RuntimeError("GPU OOM")):
            img, info = app.generate_image("crash test", "", 20, 7.0, 512, 512, -1)
            self.assertIsNone(img)
            self.assertIn("❌ Error: GPU OOM", info)


if __name__ == "__main__":
    unittest.main()
