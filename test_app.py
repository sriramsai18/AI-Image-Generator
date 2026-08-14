import unittest
from unittest.mock import MagicMock, patch
import importlib

# Pre-mocking to prevent actual Stable Diffusion model downloads or CUDA checks during testing
class TestApp(unittest.TestCase):

    def setUp(self):
        # Explicitly configure mock behavior
        self.mock_pipe = MagicMock()
        self.mock_pipe.to.return_value = self.mock_pipe

        # Mock result for calling the pipe
        self.mock_result = MagicMock()
        from PIL import Image
        self.mock_img = Image.new("RGB", (512, 512), color="red")
        self.mock_result.images = [self.mock_img]
        self.mock_pipe.return_value = self.mock_result

        # Patches for imports
        self.patcher_cuda = patch("torch.cuda.is_available", return_value=False)
        self.patcher_pretrained = patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=self.mock_pipe)
        self.patcher_url_ok = patch("gradio.networking.url_ok", return_value=True)

        self.patcher_cuda.start()
        self.patcher_pretrained.start()
        self.patcher_url_ok.start()

    def tearDown(self):
        self.patcher_cuda.stop()
        self.patcher_pretrained.stop()
        self.patcher_url_ok.stop()

    def test_app_initialization(self):
        # Reload app module to run module-level loading logic under mocked scope
        import app
        importlib.reload(app)
        self.assertIsNotNone(app.demo)
        self.assertIsNotNone(app.pipe)

    def test_generate_image_success(self):
        import app
        importlib.reload(app)

        # Test valid generation
        image, info = app.generate_image(
            prompt="a futuristic city",
            negative_prompt="blurry",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        self.assertEqual(image, self.mock_img)
        self.assertIn("Generated in", info)
        self.assertIn("Steps: 25", info)
        self.assertIn("CFG: 7.5", info)
        self.assertIn("Seed: 42", info)

    def test_generate_image_empty_prompt(self):
        import app
        importlib.reload(app)

        # Test empty prompt error
        image, info = app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )

        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first!", info)

    def test_generate_image_exception(self):
        import app
        importlib.reload(app)

        # Simulate pipe throwing an exception
        self.mock_pipe.side_effect = Exception("Model run failed")

        image, info = app.generate_image(
            prompt="valid prompt",
            negative_prompt="",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1
        )

        self.assertIsNone(image)
        self.assertIn("Error: Model run failed", info)

if __name__ == "__main__":
    unittest.main()
