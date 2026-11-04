"""Regression tests for small message and wording fixes (synthetic data only).
Run from the repository root:
    python backend/test_small_fixes.py"""
import sys, unittest, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "backend")
import engine
warnings.filterwarnings("ignore")


def _frame(labels, seed=5):
    rng = np.random.RandomState(seed)
    y = np.array(labels)
    d = pd.DataFrame({"f0": rng.randn(len(y)), "f1": rng.randn(len(y))})
    d["target"] = y
    return d.to_csv(index=False)


class SmallFixes(unittest.TestCase):
    def test_rarest_class_named_other_still_gives_two_classes(self):
        txt = _frame(["a"] * 100 + ["b"] * 60 + ["other"] * 20)
        r = engine.analyze(txt, "target", "smote")
        self.assertEqual(r["positive_label"], "other")
        self.assertEqual(len(r["full_counts"]), 2)
        self.assertNotEqual(r["positive_label"], r["negative_label"])

    def test_downgraded_folds_are_reported(self):
        labels = ["0"] * 196 + ["1"] * 4
        r = engine.analyze(_frame(labels), "target", "clustered_smote", do_classify=True,
                           classifier="erf", folds=20)
        self.assertEqual(r["classify"]["folds"], 1)
        self.assertEqual(r["classify"]["folds_downgraded_from"], 20)

    def test_snippet_states_its_assumptions(self):
        s = engine.pipeline_snippet("clustered_smote", "erf", "target", ["f0"], [])
        self.assertIn("Assumptions of this snippet", s)
        self.assertIn("0/1 with 1 = the rare class", s)
        self.assertNotIn("kept private", s)
        self.assertNotIn("my classifier", s)


if __name__ == "__main__":
    unittest.main()
