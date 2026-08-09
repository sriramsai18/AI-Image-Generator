import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestApp(unittest.TestCase):

    def setUp(self):
        # We need to mock gradio networking to prevent ValueError: When localhost is not accessible, a shareable link must be created
        self.patcher_gradio = patch("gradio.networking.url_ok", return_value=True)
        self.patcher_gradio.start()

    def tearDown(self):
        self.patcher_gradio.stop()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_init_with_cuda(self, mock_from_pretrained, mock_is_available):
        # Mock the pipeline object
        mock_pipe = MagicMock()
        # Ensure self-returning method chains return the mock itself
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Assertions
        mock_from_pretrained.assert_called_once_with(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=importlib.import_module("torch").float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        mock_pipe.to.assert_called_with("cuda")
        mock_pipe.unet.to.assert_called_with(memory_format=importlib.import_module("torch").channels_last)
        mock_pipe.vae.to.assert_called_with(memory_format=importlib.import_module("torch").channels_last)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_init_with_cpu_low_ram(self, mock_from_pretrained, mock_sysconf, mock_is_available):
        # Mock the pipeline object
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        # Mock physical RAM < 4GB
        # Page size: 4096, Phys pages: 500000 -> 4096 * 500000 = 2048000000 bytes (~1.9 GB)
        mock_sysconf.side_effect = lambda key: 4096 if key == "SC_PAGE_SIZE" else (500000 if key == "SC_PHYS_PAGES" else -1)

        import app
        importlib.reload(app)

        # Assertions
        mock_pipe.to.assert_called_with("cpu")
        mock_pipe.enable_attention_slicing.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.sysconf", create=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_init_with_cpu_high_ram(self, mock_from_pretrained, mock_sysconf, mock_is_available):
        # Mock the pipeline object
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        # Mock physical RAM >= 4GB
        # Page size: 4096, Phys pages: 2000000 -> 4096 * 2000000 = 8192000000 bytes (~7.6 GB)
        mock_sysconf.side_effect = lambda key: 4096 if key == "SC_PAGE_SIZE" else (2000000 if key == "SC_PHYS_PAGES" else -1)

        import app
        importlib.reload(app)

        # Assertions
        mock_pipe.to.assert_called_with("cpu")
        mock_pipe.enable_attention_slicing.assert_not_called()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_success(self, mock_from_pretrained, mock_is_available):
        # Mock the pipeline object
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        # Mock generator result
        from PIL import Image
        mock_image = Image.new("RGB", (100, 100))
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipe.return_value = mock_result

        # Run generate_image
        image, info = app.generate_image(
            prompt="a beautiful cat",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            width=512,
            height=512,
            seed=42
        )

        self.assertEqual(image, mock_image)
        self.assertIn("✅ Generated in", info)
        self.assertIn("Steps: 20", info)
        self.assertIn("Seed: 42", info)

        # Ensure generator was constructed correctly and inference mode was used
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args[1]
        self.assertEqual(kwargs["prompt"], "a beautiful cat")
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["num_inference_steps"], 20)
        self.assertEqual(kwargs["guidance_scale"], 7.5)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)
        self.assertIsNotNone(kwargs["generator"])

    @patch("torch.cuda.is_available", return_value=True)
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_image_empty_prompt(self, mock_from_pretrained, mock_is_available):
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.unet = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        import app
        importlib.reload(app)

        image, info = app.generate_image("", "", 20, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("⚠️ Please enter a prompt first!", info)
