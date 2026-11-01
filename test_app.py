import os
import unittest

from PIL import Image

# Ensure MOCK_MODE is enabled before importing app
os.environ["MOCK_MODE"] = "1"
import app


class TestApp(unittest.TestCase):
    def test_mock_mode_image_generation(self):
        """Test that image generation in Mock Mode works correctly and returns a PIL image."""
        prompt = "a majestic snow-capped mountain"
        negative_prompt = "cartoon"
        steps = 25
        guidance = 7.5
        width = 512
        height = 512
        seed = -1

        image, info = app.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
        )

        self.assertIsNotNone(image, "The output image should not be None")
        self.assertIsInstance(image, Image.Image, "The output must be a PIL Image instance")
        self.assertEqual(
            image.size,
            (width, height),
            f"The output image size {image.size} must match the requested resolution {(width, height)}",
        )
        self.assertIn("Generated", info, "The status info should indicate successful generation")
        self.assertIn("Steps: 25", info, "The steps in the info must match the inputs")

    def test_empty_prompt_error(self):
        """Test that empty or whitespace prompts are rejected with a warning message."""
        image, info = app.generate_image(
            prompt="   ",
            negative_prompt="blurry",
            steps=25,
            guidance=7.5,
            width=512,
            height=512,
            seed=-1,
        )

        self.assertIsNone(image, "Image should be None for empty prompts")
        self.assertIn("Please enter a prompt first", info, "Status should contain warning message")

    def test_seed_handling(self):
        """Test that setting a valid seed is reflected in the generation info."""
        prompt = "cyberpunk city"
        negative_prompt = ""
        steps = 15
        guidance = 8.0
        width = 384
        height = 384
        seed = 42

        image, info = app.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
        )

        self.assertIsInstance(image, Image.Image)
        self.assertIn("Seed: 42", info, f"Seed info must match, but got: {info}")

    def test_get_system_ram_gb(self):
        """Test that system RAM calculation returns a reasonable positive float."""
        ram_gb = app.get_system_ram_gb()
        self.assertIsInstance(ram_gb, float)
        self.assertGreater(ram_gb, 0.0)


if __name__ == "__main__":
    unittest.main()
