"""Regression test for the rare-class split guard in engine.classify_compare (synthetic data only).
Run from the repository root:
    python backend/test_rare_split_guard.py
A file with only 2 rare rows can end up with no rare rows in the held-out split. Before the guard the
run was reported as a success with meaningless scores; it must now be refused with a clear message."""
import sys, unittest, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "backend")
import engine
warnings.filterwarnings("ignore")


def _two_rare_rows():
    rng = np.random.RandomState(3)
    n = 60
    y = np.zeros(n, int)
    y[rng.choice(n, 2, replace=False)] = 1
    d = pd.DataFrame({"f0": rng.randn(n) + 1.5 * y, "f1": rng.randn(n)})
    d["target"] = np.where(y == 1, "1", "0")
    return d.to_csv(index=False)


class RareSplitGuard(unittest.TestCase):
    def test_no_rare_rows_in_test_split_is_refused_not_scored(self):
        txt = _two_rare_rows()
        try:
            r = engine.analyze(txt, "target", "clustered_smote", do_classify=True, classifier="erf")
        except ValueError as e:
            self.assertIn("rare class", str(e))
            return
        # if it does return, the held-out split must contain both classes and must not be
        # described as a multiclass target
        self.assertTrue(all(v > 0 for v in r["test_counts"].values()), r["test_counts"])
        self.assertNotIn("Multiclass", r["classify"].get("note", ""))

    def test_three_rare_rows_still_works(self):
        rng = np.random.RandomState(3)
        n = 60
        y = np.zeros(n, int)
        y[rng.choice(n, 3, replace=False)] = 1
        d = pd.DataFrame({"f0": rng.randn(n) + 1.5 * y, "f1": rng.randn(n)})
        d["target"] = np.where(y == 1, "1", "0")
        r = engine.analyze(d.to_csv(index=False), "target", "clustered_smote", do_classify=True, classifier="erf")
        self.assertTrue(r["classify"]["erf"] is not None)


if __name__ == "__main__":
    unittest.main()
