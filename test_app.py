import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher_cuda = patch("torch.cuda.is_available", return_value=False)
        cls.patcher_pipe = patch("diffusers.StableDiffusionPipeline.from_pretrained")

        cls.mock_cuda = cls.patcher_cuda.start()
        cls.mock_from_pretrained = cls.patcher_pipe.start()

        # Mock pipeline instance and chaining methods
        cls.mock_pipe_inst = MagicMock()
        cls.mock_pipe_inst.to.return_value = cls.mock_pipe_inst
        cls.mock_from_pretrained.return_value = cls.mock_pipe_inst

        # Mock image result returned by pipe
        fake_img = Image.new("RGB", (512, 512), color="red")
        cls.mock_pipe_inst.return_value.images = [fake_img]

        # Import app module dynamically after patching
        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher_pipe.stop()
        cls.patcher_cuda.stop()

    def test_app_loaded(self):
        self.assertIsNotNone(self.app)
        self.assertIsNotNone(self.app.demo)

    def test_generate_image_empty_prompt(self):
        img, status = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_valid_prompt(self):
        img, status = self.app.generate_image("a cute cat", "", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)
        self.assertIn("Seed: 42", status)

    def test_info_text_initial_value(self):
        # Verify status textbox default initial value is set
        self.assertEqual(self.app.info_text.value, "Ready to generate image.")

if __name__ == "__main__":
    unittest.main()
