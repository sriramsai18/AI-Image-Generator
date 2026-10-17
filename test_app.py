import unittest
from unittest.mock import MagicMock, patch

class TestAppAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline and CUDA check before importing app
        cls.mock_pipe = MagicMock()
        cls.mock_pipe.to.return_value = cls.mock_pipe

        cls.patcher_pretrained = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=cls.mock_pipe)
        cls.patcher_cuda = patch("torch.cuda.is_available", return_value=False)

        cls.patcher_pretrained.start()
        cls.patcher_cuda.start()

        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher_pretrained.stop()
        cls.patcher_cuda.stop()

    def test_css_contains_focus_visible_rules(self):
        css = self.app.css
        self.assertIn(":focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important;", css)
        self.assertIn("outline-offset: 2px !important;", css)

    def test_app_demo_initialized(self):
        self.assertIsNotNone(self.app.demo)

    def test_generate_image_validation(self):
        image, info = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", info)

if __name__ == "__main__":
    unittest.main()
