import unittest
from unittest.mock import MagicMock, patch


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock StableDiffusionPipeline before importing app to avoid loading heavy weights
        cls.mock_pipe = MagicMock()
        cls.mock_result = MagicMock()
        cls.mock_result.images = ["fake_pil_image"]
        cls.mock_pipe.return_value = cls.mock_result

        cls.patcher = patch(
            "diffusers.StableDiffusionPipeline.from_pretrained",
            return_value=cls.mock_pipe,
        )
        cls.patcher.start()

        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        # Ensure pipe mock side_effect and return_value are clean between tests
        self.app.pipe.side_effect = None
        self.app.pipe.return_value = self.mock_result

    def test_app_imported(self):
        self.assertIsNotNone(self.app)

    def test_generate_image_empty_prompt(self):
        img, info = self.app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first!", info)

    def test_generate_image_valid_prompt(self):
        img, info = self.app.generate_image(
            prompt="a beautiful landscape",
            negative_prompt="blurry",
            steps=20,
            guidance=7.0,
            width=512,
            height=512,
            seed=42,
        )
        self.assertEqual(img, "fake_pil_image")
        self.assertIn("✅ Generated in", info)
        self.assertIn("Seed: 42", info)

    def test_generate_image_error_handling(self):
        self.app.pipe.side_effect = Exception("CUDA out of memory")
        img, info = self.app.generate_image(
            prompt="a galaxy",
            negative_prompt="",
            steps=10,
            guidance=5.0,
            width=256,
            height=256,
            seed=-1,
        )
        self.assertIsNone(img)
        self.assertIn("❌ Error: CUDA out of memory", info)


if __name__ == "__main__":
    unittest.main()
