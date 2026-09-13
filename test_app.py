import unittest
import app

class TestAppAccessibility(unittest.TestCase):
    def test_css_contains_focus_visible_rules(self):
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #39ff14 !important;", app.css)

if __name__ == "__main__":
    unittest.main()
