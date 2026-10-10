import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestAppUIAndLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline to prevent loading large models during test
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe

        cls.pipeline_patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=cls.mock_pipe)
        cls.pipeline_patcher.start()

        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.pipeline_patcher.stop()

    def test_css_contains_focus_visible_rules(self):
        self.assertIn(":focus-visible", self.app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", self.app.css)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    def test_generate_image_valid_prompt(self):
        dummy_img = Image.new("RGB", (512, 512), color="blue")
        mock_output = MagicMock()
        mock_output.images = [dummy_img]
        self.mock_pipe.return_value = mock_output

        img, info = self.app.generate_image("a futuristic city", "", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", info)
        self.assertIn("Seed: 42", info)

if __name__ == "__main__":
    unittest.main()
