import unittest
from unittest.mock import MagicMock, patch

class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline before importing app
        cls.mock_pipe_patcher = patch("diffusers.StableDiffusionPipeline.from_pretrained")
        cls.mock_from_pretrained = cls.mock_pipe_patcher.start()
        cls.mock_pipe = MagicMock()
        cls.mock_from_pretrained.return_value = cls.mock_pipe
        cls.mock_pipe.to.return_value = cls.mock_pipe

        cls.cuda_patcher = patch("torch.cuda.is_available", return_value=False)
        cls.cuda_patcher.start()

        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.mock_pipe_patcher.stop()
        cls.cuda_patcher.stop()

    def test_app_import_and_demo(self):
        self.assertIsNotNone(self.app.demo)

    def test_focus_visible_css_present(self):
        self.assertIn(":focus-visible", self.app.css)
        self.assertIn("outline: 2px solid #e63946 !important;", self.app.css)
        self.assertIn("box-shadow: 0 0 12px rgba(230, 57, 70, 0.6) !important;", self.app.css)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image("", "negative", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

if __name__ == "__main__":
    unittest.main()
