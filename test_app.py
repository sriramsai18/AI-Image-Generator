import unittest
from unittest.mock import MagicMock, patch

# Patch StableDiffusionPipeline.from_pretrained before app is imported
mock_pipeline_obj = MagicMock()
mock_pipeline_obj.to.return_value = mock_pipeline_obj

with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=mock_pipeline_obj):
    import app


class TestApp(unittest.TestCase):
    def test_app_module_loaded(self):
        self.assertIsNotNone(app)

    def test_generate_image_empty_prompt(self):
        img, status = app.generate_image("   ", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", status)

    @patch.object(app, "pipe")
    def test_generate_image_success(self, mock_pipe):
        mock_result = MagicMock()
        mock_result.images = ["dummy_pil_image"]
        mock_pipe.return_value = mock_result

        img, status = app.generate_image("a cute cat", "blurry", 20, 7.5, 512, 512, 42)
        self.assertEqual(img, "dummy_pil_image")
        self.assertIn("Generated in", status)
        self.assertIn("Steps: 20", status)
        self.assertIn("Seed: 42", status)
        mock_pipe.assert_called_once()

    @patch.object(app, "pipe", side_effect=RuntimeError("Pipeline error"))
    def test_generate_image_exception(self, mock_pipe):
        img, status = app.generate_image("a cute cat", "", 20, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Error: Pipeline error", status)


if __name__ == "__main__":
    unittest.main()
