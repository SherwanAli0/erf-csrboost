import copy
import json
import math
import unittest

import numpy as np
from sklearn.datasets import make_classification

from rare_event_eval.labels import LabelError, LabelSpec
from rare_event_eval.reports import render_html, training_report
from rare_event_eval.training import TRAINING_SCHEMA, failure_reason_code, train_comparison

NEW_FIELDS = ("label_mapping", "predictors", "excluded_columns", "input_parsing", "split", "model_records",
              "preprocessing", "fallback_policy", "failures", "seed")


def assert_finite_json(test, value):
    """Fail on NaN or Infinity anywhere, and require standards-compliant serialization."""
    encoded = json.dumps(value, allow_nan=False)
    test.assertNotRegex(encoded, r"\b(NaN|Infinity)\b")


class TrainingReportSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(n_samples=120, n_features=5, weights=[.75, .25], random_state=7)
        cls.X = X
        cls.labels = np.where(y, "event", "clear")
        cls.context = {"feature_names": [f"f{i}" for i in range(5)], "input_sha256": "0" * 64,
                       "parser": "test fixture", "excluded_columns": [{"column": "outcome", "role": "target"}]}
        cls.result = train_comparison(X, cls.labels, models=("logistic",), positive_label="event", negative_label="clear",
                                      timeout_seconds=60, report_context=cls.context)

    def test_new_fields_are_present_and_versioned(self):
        result = self.result
        self.assertEqual(TRAINING_SCHEMA, "training-1.1")
        self.assertEqual(result["schema_version"], "training-1.1")
        for field in NEW_FIELDS:
            self.assertIn(field, result)
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        self.assertEqual(result["predictors"], self.context["feature_names"])
        self.assertEqual(result["excluded_columns"], self.context["excluded_columns"])
        self.assertEqual(result["input_sha256"], "0" * 64)

    def test_label_mapping_split_models_preprocessing_and_fallback_records(self):
        result = self.result
        mapping = result["label_mapping"]
        self.assertEqual((mapping["mode"], mapping["positive_label"], mapping["negative_label"]), ("binary", "event", "clear"))
        self.assertEqual(mapping["counts"]["input_rows"], 120)
        split = result["split"]
        self.assertEqual(split["requested"]["mode"], "stratified")
        self.assertEqual(split["requested"]["seed"], 42)
        achieved = split["achieved"]
        self.assertEqual(achieved["mode"], "stratified")
        self.assertFalse(achieved["fallback_used"])
        counts = {name: part["rows"] for name, part in achieved["partitions"].items()}
        self.assertEqual(counts, {"train": 72, "validation": 24, "test": 24})
        self.assertEqual(sum(counts.values()), 120)
        for part in achieved["partitions"].values():
            self.assertEqual(part["positive_rows"] + part["negative_rows"], part["rows"])
        record = result["model_records"]["logistic"]
        self.assertEqual((record["id"], record["version"], record["recipe_id"]),
                         ("logistic", "1", "rare_event_eval/logistic/1"))
        self.assertEqual(record["params"]["estimator"]["random_state"], 42)
        self.assertEqual(result["outcomes"]["logistic"]["model_record"], record)
        self.assertFalse(result["preprocessing"]["mixed_type_support"])
        self.assertIn("training partition only", result["preprocessing"]["fitted_on"])
        self.assertEqual(result["fallback_policy"], {
            "policy": "strict", "fallback_used": False, "description": result["fallback_policy"]["description"]})
        self.assertEqual(result["failures"], [])
        parsing = result["input_parsing"]
        self.assertEqual((parsing["rows_input"], parsing["rows_used"], parsing["rows_excluded"]), (120, 120, 0))

    def test_recorded_actual_parameters_agree_with_the_declared_record(self):
        actual = self.result["outcomes"]["logistic"]["execution"]["model_parameters"]
        declared = self.result["model_records"]["logistic"]["params"]["estimator"]
        self.assertEqual(actual["model__max_iter"], declared["max_iter"])
        self.assertEqual(actual["model__random_state"], declared["random_state"])

    def test_json_is_standards_compliant_with_no_nan_or_infinity(self):
        assert_finite_json(self, self.result)
        report = training_report(self.result)
        assert_finite_json(self, report)

    def test_a_training_1_0_result_is_still_readable(self):
        old = copy.deepcopy(self.result)
        old["schema_version"] = "training-1.0"
        for field in NEW_FIELDS + ("feature_names", "input_sha256"):
            old.pop(field, None)
        for row in old["outcomes"].values():
            row.pop("model_record", None)
        report = training_report(old)
        self.assertNotIn("training_provenance", report)
        self.assertIn("Rare-event evaluation", render_html(report))

    def test_unknown_schema_is_refused(self):
        future = copy.deepcopy(self.result)
        future["schema_version"] = "training-9.0"
        with self.assertRaises(ValueError):
            training_report(future)

    def test_html_carries_the_new_provenance_escaped(self):
        report = training_report(self.result)
        self.assertEqual(set(report["training_provenance"]), set(NEW_FIELDS) | {"schema_version"})
        html = render_html(report)
        for text in ("training_provenance", "label_mapping", "fallback_policy", "model_records"):
            self.assertIn(text, html)
        hostile = copy.deepcopy(self.result)
        hostile["label_mapping"]["positive_label"] = "<script>alert(1)</script>"
        page = render_html(training_report(hostile))
        self.assertNotIn("<script>alert(1)</script>", page)

    def test_failures_carry_stable_reason_codes_and_no_fallback_is_used(self):
        groups = np.array(["one"] * 120)
        result = train_comparison(self.X, self.labels, models=("logistic",), positive_label="event", negative_label="clear",
                                  split_mode="group", groups=groups, timeout_seconds=60)
        row = result["outcomes"]["logistic"]
        self.assertNotEqual(row["status"], "success")
        self.assertEqual(row["reason_code"], "SPLIT_INFEASIBLE")
        self.assertEqual(result["failures"][0]["reason_code"], "SPLIT_INFEASIBLE")
        self.assertEqual(result["failures"][0]["model"], "logistic")
        self.assertIsNone(result["split"]["achieved"])
        self.assertFalse(result["fallback_policy"]["fallback_used"])
        assert_finite_json(self, result)

    def test_reason_code_classification(self):
        cases = (({"status": "timeout"}, "MODEL_TIMEOUT"),
                 ({"status": "error", "error": "Worker exited without a result."}, "WORKER_NO_RESULT"),
                 ({"status": "error", "error": "Unknown model name."}, "MODEL_UNKNOWN"),
                 ({"status": "error", "error": "This model requires optional training dependencies. Install the package with [training]."},
                  "MODEL_DEPENDENCY_MISSING"),
                 ({"status": "error", "error": "Requested training-only calibration folds are infeasible; no fallback was used."},
                  "CALIBRATION_INFEASIBLE"),
                 ({"status": "error", "error": "SMOTE requires at least two training rows in the smaller class."},
                  "RESAMPLING_INFEASIBLE"),
                 ({"status": "error", "error": "Every partition must contain both classes. Requested split is infeasible; no fallback was used."},
                  "SPLIT_INFEASIBLE"),
                 ({"status": "error", "error": "Invalid numeric features or explicit binary labels."}, "INPUT_INVALID"),
                 ({"status": "error", "error_type": "MemoryError", "error": ""}, "MODEL_OUT_OF_MEMORY"),
                 ({"status": "error", "error": "something else"}, "MODEL_FIT_ERROR"))
        for payload, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(failure_reason_code(payload), expected)

    def test_only_the_strict_fallback_policy_exists(self):
        with self.assertRaises(ValueError):
            train_comparison(self.X, self.labels, models=("logistic",), positive_label="event", negative_label="clear",
                             fallback_policy="random_oversample")


class TrainingLabelHandling(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=120, n_features=5, weights=[.75, .25], random_state=7)
        self.X = X
        self.labels = np.where(y, "event", "clear")

    def test_label_errors_are_raised_before_any_worker_starts(self):
        for labels, kwargs, code in ((np.array(["event"] * 120), {"positive_label": "event", "negative_label": "clear"}, "LABEL_SINGLE_CLASS"),
                                     (self.labels, {"positive_label": "missing", "negative_label": "clear"}, "LABEL_POSITIVE_ABSENT"),
                                     (self.labels, {"negative_label": "clear"}, "LABEL_POSITIVE_REQUIRED")):
            with self.subTest(code=code), self.assertRaises(LabelError) as caught:
                train_comparison(self.X, labels, models=("logistic",), **kwargs)
            self.assertEqual(caught.exception.code, code)

    def test_missing_targets_raise_unless_drop_reported_and_are_then_listed(self):
        labels = self.labels.astype(object)
        labels[5] = ""
        with self.assertRaises(LabelError) as caught:
            train_comparison(self.X, labels, models=("logistic",), positive_label="event", negative_label="clear")
        self.assertEqual(caught.exception.code, "LABEL_MISSING_TARGET")
        result = train_comparison(self.X, labels, models=("logistic",), label_spec=LabelSpec("event", "clear"),
                                  missing_policy="drop_reported", timeout_seconds=60)
        parsing = result["input_parsing"]
        self.assertEqual((parsing["rows_input"], parsing["rows_used"], parsing["rows_excluded"]), (120, 119, 1))
        self.assertEqual(parsing["row_exclusions"][0]["row_index"], 5)
        self.assertEqual(result["label_mapping"]["counts"]["missing_target_rows"], 1)
        self.assertEqual(result["split"]["achieved"]["rows"], 119)
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        assert_finite_json(self, result)

    def test_one_vs_rest_trains_on_the_binary_view_and_keeps_the_original_mapping(self):
        labels = self.labels.astype(object)
        labels[np.flatnonzero(labels == "clear")[::2]] = "other"
        result = train_comparison(self.X, labels, models=("logistic",), label_spec=LabelSpec("event", mode="one_vs_rest"),
                                  timeout_seconds=60)
        mapping = result["label_mapping"]
        self.assertEqual(mapping["mode"], "one_vs_rest")
        self.assertEqual(mapping["negative_labels"], ["clear", "other"])
        self.assertEqual({item["label"] for item in mapping["original_labels"]}, {"clear", "event", "other"})
        outcome = result["outcomes"]["logistic"]
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(outcome["report"]["task"]["positive_label"], "event")
        self.assertEqual(result["input_parsing"]["rows_used"], 120)
        with self.assertRaises(LabelError) as caught:
            train_comparison(self.X, labels, models=("logistic",), positive_label="event", negative_label="clear")
        self.assertEqual(caught.exception.code, "LABEL_MULTICLASS_REQUIRES_OVR")


if __name__ == "__main__":
    unittest.main()
