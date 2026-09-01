import unittest

class TestAppAccessibility(unittest.TestCase):
    def test_css_contains_focus_visible_rules(self):
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn(":focus-visible", content)
        self.assertIn(".gradio-container button:focus-visible", content)
        self.assertIn("outline: 2px solid #e63946 !important;", content)

if __name__ == "__main__":
    unittest.main()
