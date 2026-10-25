"""Smoke-test the Streamlit UI in demo mode with Streamlit's AppTest. Skipped without Streamlit."""
import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def button(at, label):
    return next(b for b in at.button if b.label == label)


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class AppSmokeTests(unittest.TestCase):
    def fresh(self):
        return AppTest.from_file(APP, default_timeout=60).run()

    def test_renders_in_demo_mode_without_exceptions(self):
        at = self.fresh()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertEqual(at.title[0].value, "GraphCite")

    def test_two_hop_demo_question_is_answered_with_verified_citations(self):
        at = self.fresh()
        at.text_input(key="question").set_value("Who founded the company that develops Helios Drive?").run()
        button(at, "Answer with citations").click().run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertIn("Maria Santos", " ".join(m.value for m in at.markdown))
        self.assertTrue(any(m.label == "Claims supported by their citations" and m.value == "100%" for m in at.metric))

    def test_unknown_topic_is_reported_not_hallucinated(self):
        at = self.fresh()
        at.text_input(key="question").set_value("What is the capital of Portugal?").run()
        button(at, "Answer with citations").click().run()
        self.assertIn("nothing to answer", " ".join(m.value for m in at.markdown))

    def test_empty_question_shows_error(self):
        at = self.fresh()
        button(at, "Answer with citations").click().run()
        self.assertIn("Enter a question", at.error[0].value)


if __name__ == "__main__":
    unittest.main()
