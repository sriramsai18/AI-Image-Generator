import unittest
import app

class TestAppAccessibility(unittest.TestCase):
    def test_custom_css_contains_focus_visible(self):
        self.assertIn(":focus-visible", app.css)
        self.assertIn("outline: 2px solid #00f0ff !important;", app.css)
        self.assertIn("outline-offset: 2px !important;", app.css)
        self.assertIn("box-shadow: 0 0 10px rgba(0,240,255,0.6) !important;", app.css)

    def test_demo_interface_created(self):
        self.assertIsNotNone(app.demo)
        self.assertEqual(app.demo.title, "Text2Image — Sriram")

if __name__ == "__main__":
    unittest.main()
