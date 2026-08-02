import unittest
from unittest.mock import MagicMock, patch


# We need to mock StableDiffusionPipeline BEFORE importing app in tests
class MockPipeline:
    def __init__(self):
        self.unet = MagicMock()
        self.vae = MagicMock()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, *args, **kwargs):
        return self

    def enable_attention_slicing(self):
        pass

    def __call__(self, *args, **kwargs):
        mock_image_obj = MagicMock()
        mock_image_obj.images = [MagicMock()]
        return mock_image_obj

with patch("diffusers.StableDiffusionPipeline.from_pretrained", MockPipeline.from_pretrained), \
     patch("torch.cuda.is_available", return_value=False):
    import app

class TestAppOptimizations(unittest.TestCase):

    def test_get_system_ram_posix(self):
        # When mocking os.sysconf, use create=True for cross-platform safety
        with patch("os.sysconf", create=True) as mock_sysconf:
            mock_sysconf.side_effect = lambda key: 4096 if key == "SC_PAGE_SIZE" else (1024 * 1024 if key == "SC_PHYS_PAGES" else 0)
            ram = app.get_system_ram()
            self.assertEqual(ram, 4096 * 1024 * 1024)  # 4 GB

    def test_get_system_ram_fallback(self):
        with patch("os.sysconf", side_effect=ValueError, create=True):
            ram = app.get_system_ram()
            self.assertEqual(ram, 8 * 1024**3)  # default 8 GB fallback

    def test_generate_image_empty_prompt(self):
        img, info = app.generate_image("", "", 25, 7.5, 512, 512, -1)
        self.assertIsNone(img)
        self.assertIn("Please enter a prompt first", info)

    def test_generate_image_success(self):
        with patch("app.pipe", MockPipeline()):
            img, info = app.generate_image("test prompt", "", 25, 7.5, 512, 512, 42)
            self.assertIsNotNone(img)
            self.assertIn("✅ Generated in", info)

if __name__ == "__main__":
    unittest.main()
