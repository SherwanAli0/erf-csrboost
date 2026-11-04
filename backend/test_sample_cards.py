"""The bundled demo cards must describe the class the app actually treats as the rare (positive) one.
For each bundled sample this checks that the class the app picks as positive has the expected size,
that the row count and ratio quoted in the card text match the file, and that the card text names the
right class. It also checks the demo noun used by the page. Bundled sample data only.
Run: py -3.11 backend/test_sample_cards.py"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, "backend")
import engine  # noqa: E402
import server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
INDEX = ROOT / "frontend" / "index.html"

# expected: (rows, rare rows with target 1, words the card text must contain, phrases it must not contain)
EXPECTED = {
    "liver_disease.csv":   (583, 167, ["without liver disease"], ["spot patients", "patients with liver disease"]),
    "credit_default.csv":  (6000, 1327, ["defaults"], []),
    "student_dropout.csv": (4424, 794, ["enrolled"], ["at risk", "dropping out"]),
    "car_quality.csv":     (1728, 69, ["good"], []),
}


class SampleCardTests(unittest.TestCase):
    def test_every_bundled_sample_has_a_card_and_an_expectation(self):
        self.assertEqual(set(server.SAMPLE_META), set(EXPECTED))
        for name in EXPECTED:
            self.assertTrue((SAMPLES / name).exists(), name)

    def test_positive_class_and_counts_match_the_card(self):
        for name, (rows, rare, must, banned) in EXPECTED.items():
            with self.subTest(sample=name):
                r = engine.analyze((SAMPLES / name).read_text(encoding="utf-8"), "target",
                                   "clustered_smote", do_classify=False)
                fc = r["full_counts"]
                pos, neg = r["positive_label"], r["negative_label"]
                # the app picks the rare class (target 1) as the positive one
                self.assertEqual(pos, "1")
                self.assertEqual(fc[pos], rare)
                self.assertEqual(fc[pos] + fc[neg], rows)
                self.assertLess(fc[pos], fc[neg])
                desc = server.SAMPLE_META[name]["desc"]
                m = re.search(r"\(([\d,]+) rows, ([\d.]+):1\)", desc)
                self.assertIsNotNone(m, desc)
                self.assertEqual(int(m.group(1).replace(",", "")), rows, desc)
                self.assertEqual(float(m.group(2)), round(fc[neg] / fc[pos], 1), desc)
                for w in must:
                    self.assertIn(w, desc.lower(), desc)
                for w in banned:
                    self.assertNotIn(w, desc.lower(), desc)

    def test_demo_noun_does_not_call_the_liver_rare_class_patients(self):
        html = INDEX.read_text(encoding="utf-8")
        m = re.search(r"const DEMO_NOUN=\{([^}]*)\}", html)
        self.assertIsNotNone(m)
        nouns = dict(re.findall(r"'([\w.]+)':'(\w+)'", m.group(1)))
        self.assertEqual(set(nouns), set(EXPECTED))
        self.assertEqual(nouns["liver_disease.csv"], "records")
        self.assertEqual(nouns["student_dropout.csv"], "students")


if __name__ == "__main__":
    unittest.main()
