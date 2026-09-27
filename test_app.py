import unittest
from unittest.mock import patch, MagicMock

class TestAppUXAndAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = patch('diffusers.StableDiffusionPipeline.from_pretrained')
        cls.mock_from_pretrained = cls.patcher.start()

        # Configure mock pipeline return value
        cls.mock_pipe = MagicMock()
        cls.mock_from_pretrained.return_value = cls.mock_pipe
        cls.mock_pipe.to.return_value = cls.mock_pipe

        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_focus_visible_css_present(self):
        """Verify that high-contrast focus-visible styles exist for keyboard accessibility."""
        css = self.app.css
        self.assertIn(".gradio-container button:focus-visible", css)
        self.assertIn(".gradio-container input:focus-visible", css)
        self.assertIn(".gradio-container textarea:focus-visible", css)
        self.assertIn(".gradio-container a:focus-visible", css)
        self.assertIn("outline: 2px solid #e63946 !important", css)

    def test_footer_links_rel_noopener(self):
        """Verify external footer links include security and accessibility attributes."""
        # Find HTML blocks in demo
        found_rel = False
        for block in self.app.demo.blocks.values():
            if hasattr(block, 'value') and isinstance(block.value, str):
                if 'app-footer' in block.value:
                    self.assertIn('rel="noopener noreferrer"', block.value)
                    found_rel = True
        self.assertTrue(found_rel, "Footer HTML block with rel='noopener noreferrer' not found")

    def test_generate_image_empty_prompt(self):
        """Verify generate_image handles empty or whitespace-only prompts gracefully."""
        image, status = self.app.generate_image("", "blurry", 25, 7.5, 512, 512, -1)
        self.assertIsNone(image)
        self.assertIn("Please enter a prompt first", status)

    def test_generate_image_valid_prompt(self):
        """Verify generate_image calls the model pipeline and returns success status."""
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        self.mock_pipe.return_value = mock_result

        image, status = self.app.generate_image("a cyberpunk neon city", "blurry", 25, 7.5, 512, 512, 42)
        self.assertIsNotNone(image)
        self.assertIn("Generated in", status)
        self.assertIn("Seed: 42", status)

if __name__ == '__main__':
    unittest.main()
