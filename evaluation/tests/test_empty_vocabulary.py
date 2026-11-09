"""A categorical predictor with no observed value in the training partition (empty vocabulary).

SMOTENC cannot oversample such a table. The failure must name the real cause with a dedicated stable reason
code (``FEATURE_EMPTY_CATEGORY_IN_TRAIN``), stay strict (status ``failed``, no fallback sampler), and leave every
recipe that does not call SMOTENC, and every non-empty vocabulary, exactly as before.
"""
import unittest
from unittest import mock

import numpy as np

from mixed_support import assert_finite_json, mixed_table, patched_runner
from rare_event_eval.labels import LabelSpec
from rare_event_eval.mixed import train_comparison_tabular
from rare_event_eval.mixed_models import FEATURE_EMPTY_CATEGORY_IN_TRAIN, RecipeError, TabularPreprocessor, resample_mixed
from rare_event_eval.splits import make_split

FRAUD = LabelSpec("fraud", "ok")


def with_blank_training_region():
    """A table whose ``region`` column is blank on every training row and filled elsewhere."""
    columns = mixed_table()
    y = np.array([value == "fraud" for value in columns["label"]]).astype(int)
    split = make_split(y, mode="stratified", seed=42)
    train = set(split.train.tolist())
    columns["region"] = ["" if i in train else ("north" if i % 2 else "south") for i in range(len(y))]
    return columns


class UnitLevel(unittest.TestCase):
    def pre_and_rows(self, empty=True):
        rng = np.random.RandomState(0)
        n = 80
        y = np.zeros(n, dtype=int)
        y[:12] = 1
        rows = np.empty((n, 3), dtype=object)
        rows[:, 0] = rng.normal(size=n) + y
        rows[:, 1] = rng.choice(["a", "b", "c"], n)
        rows[:, 2] = "" if empty else rng.choice(["x", "y"], n)
        pre = TabularPreprocessor(numeric_columns=(0,), categorical_columns=(1, 2),
                                  column_names=("x", "city", "region")).fit(rows)
        return pre, pre.encode(rows), y

    def test_empty_vocabulary_gets_its_own_code_message_and_details(self):
        pre, coded, y = self.pre_and_rows()
        self.assertEqual(pre.cardinalities_, [3, 0])
        with mock.patch("imblearn.over_sampling.SMOTENC") as sampler:
            with self.assertRaises(RecipeError) as raised:
                resample_mixed(coded, y, pre, method="smote", retention=.5, seed=7)
        sampler.assert_not_called()  # rejected before any sampler is built
        error = raised.exception
        self.assertEqual(error.code, "FEATURE_EMPTY_CATEGORY_IN_TRAIN")
        self.assertEqual(error.code, FEATURE_EMPTY_CATEGORY_IN_TRAIN)
        self.assertEqual(error.details, {"columns": ["region"], "column_count": 1})
        message = str(error)
        self.assertIn("'region'", message)
        self.assertIn("no observed value in the training partition", message)
        self.assertIn("No other sampler was substituted", message)
        self.assertNotIn("outside the learned vocabulary", message)
        self.assertNotEqual(error.code, "RESAMPLING_INVALID_CATEGORY_CODE")

    def test_clustered_recipe_reaches_the_same_failure(self):
        pre, coded, y = self.pre_and_rows()
        with self.assertRaises(RecipeError) as raised:
            resample_mixed(coded, y, pre, method="clustered_smote", retention=.5, seed=7)
        self.assertEqual(raised.exception.code, "FEATURE_EMPTY_CATEGORY_IN_TRAIN")

    def test_clustering_without_smote_is_not_affected(self):
        pre, coded, y = self.pre_and_rows()
        Xr, yr, event = resample_mixed(coded, y, pre, method="clustered", retention=.5, seed=7)
        self.assertEqual(event["method"], "clustered")
        self.assertEqual(len(Xr), len(yr))

    def test_many_empty_columns_are_listed_with_a_bounded_message(self):
        rng = np.random.RandomState(1)
        n = 60
        y = np.zeros(n, dtype=int)
        y[:9] = 1
        rows = np.empty((n, 8), dtype=object)
        rows[:, 0] = rng.normal(size=n)
        for position in range(1, 8):
            rows[:, position] = ""
        pre = TabularPreprocessor(numeric_columns=(0,), categorical_columns=tuple(range(1, 8)),
                                  column_names=tuple(f"c{i}" for i in range(8))).fit(rows)
        with self.assertRaises(RecipeError) as raised:
            resample_mixed(pre.encode(rows), y, pre, method="smote", retention=1.0, seed=3)
        self.assertEqual(raised.exception.details["column_count"], 7)
        self.assertEqual(raised.exception.details["columns"], [f"c{i}" for i in range(1, 8)])
        self.assertIn("and 2 more", str(raised.exception))

    def test_non_empty_vocabularies_are_unchanged(self):
        pre, coded, y = self.pre_and_rows(empty=False)
        Xr, yr, event = resample_mixed(coded, y, pre, method="smote", retention=.5, seed=7)
        self.assertEqual(event["smote_variant"], "SMOTENC")
        self.assertTrue(event["synthetic_category_codes_in_vocabulary"])
        self.assertEqual(event["output_counts"][0], event["output_counts"][1])

    def test_a_one_value_vocabulary_is_still_valid(self):
        rng = np.random.RandomState(2)
        n = 80
        y = np.zeros(n, dtype=int)
        y[:12] = 1
        rows = np.empty((n, 2), dtype=object)
        rows[:, 0] = rng.normal(size=n)
        rows[:, 1] = "only"
        pre = TabularPreprocessor(numeric_columns=(0,), categorical_columns=(1,), column_names=("x", "kind")).fit(rows)
        self.assertEqual(pre.cardinalities_, [1])
        _, _, event = resample_mixed(pre.encode(rows), y, pre, method="smote", retention=1.0, seed=5)
        self.assertEqual(event["smote_variant"], "SMOTENC")

    def test_balanced_classes_need_no_sampler_and_do_not_fail(self):
        pre, coded, _ = self.pre_and_rows()
        y = np.array([0, 1] * 40)
        Xr, yr, event = resample_mixed(coded, y, pre, method="smote", retention=1.0, seed=7)
        self.assertEqual(event["synthetic_rows"], 0)
        self.assertEqual(len(Xr), 80)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.columns = with_blank_training_region()
        runner, patch = patched_runner()
        with patch:
            cls.result = train_comparison_tabular(
                cls.columns, target="label", label_spec=FRAUD,
                models=("logistic", "smote_rf", "csrboost", "erf_one_bag", "erf_no_smote"))

    def test_the_column_really_has_an_empty_training_vocabulary(self):
        record = self.result["preprocessing"]["record"]["categorical_columns"]["region"]
        self.assertEqual(record["vocabulary_size"], 0)
        self.assertEqual(record["indicator_columns"], [])
        self.assertGreater(record["unseen_category_cells"]["test"], 0)

    def test_sampling_recipes_fail_strictly_with_the_accurate_code(self):
        for name in ("smote_rf", "csrboost", "erf_one_bag"):
            row = self.result["outcomes"][name]
            self.assertEqual((row["status"], row["reason_code"]), ("failed", "FEATURE_EMPTY_CATEGORY_IN_TRAIN"), name)
            self.assertIn("'region'", row["error"])
            self.assertIn("no observed value in the training partition", row["error"])
            self.assertNotIn("outside the learned vocabulary", row["error"])
            self.assertNotIn("report", row)
        failures = {item["model"]: item for item in self.result["failures"]}
        self.assertEqual(sorted(failures), ["csrboost", "erf_one_bag", "smote_rf"])
        for item in failures.values():
            self.assertEqual(item["reason_code"], "FEATURE_EMPTY_CATEGORY_IN_TRAIN")
        policy = self.result["fallback_policy"]
        self.assertEqual((policy["policy"], policy["fallback_used"]), ("strict", False))
        self.assertNotIn("RESAMPLING_INVALID_CATEGORY_CODE", str(self.result))
        assert_finite_json(self, self.result)

    def test_recipes_without_smote_still_succeed(self):
        for name in ("logistic", "erf_no_smote"):
            self.assertEqual(self.result["outcomes"][name]["status"], "success", name)
            self.assertNotIn("reason_code", self.result["outcomes"][name])


if __name__ == "__main__":
    unittest.main()
