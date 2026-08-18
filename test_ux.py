import unittest
import re

class TestUXAccessibility(unittest.TestCase):
    def test_css_contains_focus_visible_styles(self):
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        match = re.search(r'css\s*=\s*"""(.*?)"""', content, re.DOTALL)
        self.assertIsNotNone(match, "CSS string not found in app.py")

        css_content = match.group(1)
        self.assertIn("focus-visible", css_content)
        self.assertIn("outline: 2px solid #e63946 !important;", css_content)
        self.assertIn("outline-offset: 2px !important;", css_content)
        self.assertIn(".gradio-container button:focus-visible", css_content)

if __name__ == "__main__":
    unittest.main()
