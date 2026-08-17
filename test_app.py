import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

class TestAppOptimizations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create mock pipeline and objects before importing app
        cls.mock_pipeline_cls = patch('diffusers.StableDiffusionPipeline.from_pretrained').start()
        cls.mock_pipe = MagicMock()
        cls.mock_pipeline_cls.return_value = cls.mock_pipe
        cls.mock_pipe.to.return_value = cls.mock_pipe

        # Mock generated output image
        mock_output = MagicMock()
        mock_output.images = [Image.new('RGB', (512, 512), color='blue')]
        cls.mock_pipe.return_value = mock_output

        # Import app module
        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        patch.stopall()

    def setUp(self):
        self.app.pipe.reset_mock()

    def test_generate_image_empty_prompt(self):
        img, status = self.app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_success(self):
        # Configure return value explicitly for mock_pipe call
        mock_output = MagicMock()
        mock_output.images = [Image.new('RGB', (512, 512), color='blue')]
        self.app.pipe.return_value = mock_output
        self.app.pipe.side_effect = None

        img, status = self.app.generate_image("a beautiful sunset", "blurry", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(img)
        self.assertIn("Generated in", status)
        self.assertIn("Seed: 42", status)

        # Verify pipeline call
        self.app.pipe.assert_called_once()

    def test_generate_image_error_handling(self):
        self.app.pipe.side_effect = Exception("CUDA out of memory")
        img, status = self.app.generate_image("a cat", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: CUDA out of memory", status)
        self.app.pipe.side_effect = None  # Reset side effect

if __name__ == '__main__':
    unittest.main()
