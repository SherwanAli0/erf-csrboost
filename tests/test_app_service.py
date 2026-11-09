"""Parity and boundary tests for the application service layer (``backend/services``).

What this file establishes, offline and deterministically, with small inline fixtures:

* (a) metric parity between the legacy engine metric functions and ``rare_event_eval.evaluate_predictions``
  on identical fixed predictions and thresholds, with the definition differences asserted;
* (b) one mixed-type table through the legacy engine and through the service, with the expected recipe
  differences asserted explicitly;
* (c) mixed-input regression: every table-shaped input accepted by ``backend/test_torture.py`` and the four
  bundled samples still profiles and trains through the service, and each intentional difference is asserted;
* (d) leakage boundary through the service: no train-fitted transform sees validation or test rows;
* (e) report JSON holds no NaN or Infinity and labels are escaped in the HTML report.

Run from the repository root with an interpreter that has the evaluation package importable (an editable
install, or ``PYTHONPATH=evaluation/src``; this file adds the source folder itself when needed)::

    python -B -m unittest discover -s tests -v

Tests that need the legacy engine (pandas, imbalanced-learn) or openpyxl are skipped, with a reason, in
environments that lack them. The live server is never started except in one test that checks the service
works beside it.
"""
from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
import warnings
from datetime import date, timedelta
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
PACKAGE_SOURCE = ROOT / "evaluation" / "src"
PACKAGE_TESTS = ROOT / "evaluation" / "tests"
SAMPLES = ROOT / "samples"

if importlib.util.find_spec("rare_event_eval") is None:  # worker subprocesses inherit PYTHONPATH
    sys.path.insert(0, str(PACKAGE_SOURCE))
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE_SOURCE), os.environ.get("PYTHONPATH", "")]))
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(PACKAGE_TESTS))

from rare_event_eval import CostAssumptions, evaluate_predictions  # noqa: E402
from rare_event_eval.policies import select_threshold  # noqa: E402
from rare_event_eval.registry import MIXED_REGISTRY, REGISTRY  # noqa: E402
from rare_event_eval.splits import make_split  # noqa: E402
from mixed_support import InProcessRunner  # noqa: E402

import services  # noqa: E402
from services import (ServiceError, evaluate_saved_predictions, legacy_recipe_map, profile_for_app,  # noqa: E402
                      render_report_html, train_for_app)

try:
    import pandas  # noqa: F401
    import engine as legacy
    LEGACY_REASON = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    legacy = None
    LEGACY_REASON = f"the legacy engine is not importable here ({type(exc).__name__})"
try:
    import openpyxl  # noqa: F401
    OPENPYXL = True
except ImportError:  # pragma: no cover
    OPENPYXL = False

needs_legacy = unittest.skipIf(legacy is None, LEGACY_REASON or "")
needs_legacy_readers = unittest.skipIf(legacy is None or not OPENPYXL,
                                       LEGACY_REASON or "openpyxl is needed for the Excel torture input")


def strict_json(test, value):
    """Serialize with allow_nan=False and parse back rejecting bare NaN or Infinity constants."""
    def reject(token):
        raise AssertionError(f"bare JSON constant {token}")
    text = json.dumps(value, allow_nan=False)
    json.loads(text, parse_constant=reject)
    return text


# ---------------------------------------------------------------------------------------------- fixtures
def fixed_predictions():
    """60 rows, 12 positives, probabilities rounded to four decimals (so some tie, none is round)."""
    rng = np.random.RandomState(3)
    y = np.array([1] * 12 + [0] * 48)
    rng.shuffle(y)
    scores = np.round(np.where(y == 1, rng.beta(4, 2, 60), rng.beta(2, 4, 60)), 4)
    return y, scores


def package_evaluation(y, scores, threshold):
    labels = np.where(y == 1, "pos", "neg").tolist()
    return evaluate_predictions(labels, {"m": scores.tolist()}, positive_label="pos", negative_label="neg",
                                score_kind="probability", threshold=threshold)["models"]["m"]


def mixed_fixture_csv(n=300, seed=11):
    """Numeric (with missing cells), categorical (with a missing spelling) and a 0/1 target at about 14 percent."""
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < 0.14).astype(int)
    lines = ["amount,age,city,segment,target"]
    for i in range(n):
        amount = "" if i % 11 == 0 else f"{abs(rng.randn()) * 100 + 60 * y[i]:.2f}"
        age = f"{rng.randint(18, 80)}"
        city = "?" if i % 17 == 0 else str(rng.choice(["Berlin", "Munich", "Hamburg"]))
        segment = "retail" if rng.rand() < 0.35 + 0.3 * y[i] else "business"
        lines.append(f"{amount},{age},{city},{segment},{y[i]}")
    return "\n".join(lines) + "\n"


def dict_table(n=140, seed=3, positives=28, row_ids=False, labels=("fraud", "ok"), extra=None):
    """Column dict of strings (blank = missing): numeric, categorical and a label column."""
    rng = np.random.RandomState(seed)
    y = np.zeros(n, dtype=bool)
    y[rng.choice(n, positives, replace=False)] = True
    columns = {}
    if row_ids:
        columns["rid"] = [str(1000 + 7 * i) for i in range(n)]  # not 0..n, so no reader mistakes it for a row index
    columns["age"] = ["" if i % 19 == 0 else f"{v:.3f}" for i, v in enumerate(rng.normal(40, 10, n) + 8 * y)]
    columns["income"] = [f"{v:.3f}" for v in rng.gamma(3, 10, n) - 5 * y]
    columns["color"] = ["" if i % 23 == 0 else str(v) for i, v in enumerate(
        np.where(rng.rand(n) < 0.35 + 0.3 * y, "red", rng.choice(["green", "blue"], n)))]
    columns["city"] = [str(v) for v in rng.choice(["a", "b", "c", "d"], n)]
    for name, values in (extra or {}).items():
        columns[name] = values
    columns["label"] = [labels[0] if flag else labels[1] for flag in y]
    return columns


def to_csv_bytes(columns):
    names = list(columns)
    rows = zip(*(columns[name] for name in names))
    return ("\n".join([",".join(names)] + [",".join(row) for row in rows]) + "\n").encode("utf-8")


def torture_inputs():
    """The exact inputs of ``backend/test_torture.py`` (its case-building half, without running the models)."""
    source = (BACKEND / "test_torture.py").read_text(encoding="utf-8").split("passed = failed = 0", 1)[0]
    namespace = {"__name__": "torture_inputs"}
    exec(compile(source, "test_torture_inputs", "exec"), namespace)  # noqa: S102 - our own repository file
    inputs = {key: (value.encode("utf-8"), "") for key, value in namespace["CASES"].items()}
    inputs.update({key: (value, key) for key, value in namespace["BYTE_CASES"].items()})
    return inputs, namespace["MUST_REFUSE"] | namespace["BYTE_MUST_REFUSE"]


def confirmed_request(profile):
    """The request a user would send after confirming the profile's suggestions (the service never infers them)."""
    classes = profile["classes"]
    labels = [item["label"] for item in classes["labels"]]
    positive = classes["suggested_positive"]["label"]
    request = {"target": profile["target"]["column"], "positive_label": positive,
               "exclude_columns": [item["column"] for item in profile["suggested_exclusions"]]}
    if classes["suggested_label_mode"] == "binary":
        request["negative_label"] = next(label for label in labels if label != positive)
    else:
        request["one_vs_rest"] = True
    return request


# ---------------------------------------------------------------------------------------------- typed errors
class ServiceErrors(unittest.TestCase):
    def test_codes_are_stable_unique_strings(self):
        codes = services.APP_ERROR_CODES
        self.assertEqual(len(codes), len(set(codes)))
        self.assertTrue(all(code.startswith("APP_") and code.isupper() for code in codes))

    def test_messages_have_no_paths_or_tracebacks(self):
        windows, unix = "C:" + chr(92) + "Users" + chr(92) + "someone" + chr(92) + "data.csv", "/home/someone/data/x.csv"
        for raw in (f"cannot open {windows} now", f"cannot open {unix} now", "line one\nTraceback (most recent call last)"):
            text = services.safe_text(raw)
            self.assertNotIn("someone", text)
            self.assertNotIn("Traceback", text)
        self.assertIn("<path>", services.safe_text(f"cannot open {windows} now"))

    def test_unexpected_exception_becomes_a_generic_typed_error(self):
        secret = "C:" + chr(92) + "Users" + chr(92) + "private" + chr(92) + "file.csv"
        with mock.patch("rare_event_eval.evaluate_predictions", side_effect=OSError(f"cannot read {secret}")):
            with self.assertRaises(ServiceError) as caught:
                evaluate_saved_predictions(["a", "b"], {"m": [0.1, 0.9]}, positive_label="a", negative_label="b",
                                           score_kind="probability")
        error = caught.exception
        self.assertEqual(error.code, "APP_INTERNAL_ERROR")
        self.assertNotIn("private", str(error))
        self.assertNotIn("Traceback", json.dumps(error.to_record()))

    def test_invalid_saved_predictions_raise_a_typed_error_with_a_package_message(self):
        with self.assertRaises(ServiceError) as caught:
            evaluate_saved_predictions(["pos", "neg"], {"m": [0.5, float("nan")]}, positive_label="pos",
                                       negative_label="neg", score_kind="probability")
        self.assertEqual((caught.exception.code, caught.exception.layer), ("APP_EVALUATION_INVALID", "package"))
        self.assertIn("finite", str(caught.exception))
        strict_json(self, caught.exception.to_record())

    def test_package_errors_keep_their_stable_codes(self):
        columns = dict_table()
        with self.assertRaises(ServiceError) as caught:
            train_for_app(columns, target="label", positive_label="absent", negative_label="ok", models=("logistic",))
        self.assertEqual((caught.exception.code, caught.exception.layer), ("LABEL_POSITIVE_ABSENT", "package"))
        with self.assertRaises(ServiceError) as caught:
            train_for_app(columns, target="label", positive_label="fraud")
        self.assertEqual(caught.exception.code, "APP_INPUT_INVALID")
        with self.assertRaises(ServiceError) as caught:
            train_for_app(columns, target="label", positive_label="fraud", negative_label="ok", models=("not_a_model",))
        self.assertEqual(caught.exception.code, "OPTION_INVALID")

    def test_whole_request_limits_are_typed_before_any_model_runs(self):
        from rare_event_eval.tabular import TabularLimits
        runner = InProcessRunner()
        columns = dict_table(extra={"name": [f"user{i}" for i in range(140)]})
        with mock.patch("rare_event_eval.mixed.run_worker_directory", runner):
            with self.assertRaises(ServiceError) as caught:
                train_for_app(columns, target="label", positive_label="fraud", negative_label="ok", models=("logistic",))
            self.assertEqual(caught.exception.code, "TABLE_CARDINALITY_EXCEEDED")
            with self.assertRaises(ServiceError) as caught:
                train_for_app(to_csv_bytes(dict_table()), target="label", positive_label="fraud", negative_label="ok",
                              limits=TabularLimits(max_bytes=100))
            self.assertEqual(caught.exception.code, "TABLE_TOO_LARGE_BYTES")
        self.assertEqual(runner.calls, [])

    @needs_legacy
    def test_unsupported_upload_formats_are_refused_with_a_typed_error(self):
        for name, blob in (("data.parquet", b"PAR1" + b"\x00" * 100), ("data.sav", b"$FL2@(#) SPSS DATA FILE" + b"\x00" * 50)):
            with self.subTest(name=name):
                with self.assertRaises(ServiceError) as caught:
                    profile_for_app(blob, name)
                self.assertEqual((caught.exception.code, caught.exception.layer), ("APP_FORMAT_UNSUPPORTED", "legacy_input"))

    def test_missing_target_suggestion_without_a_dataframe_needs_an_explicit_target(self):
        with self.assertRaises(ServiceError) as caught:
            profile_for_app(dict_table())
        self.assertEqual(caught.exception.code, "APP_TARGET_REQUIRED")
        profile = profile_for_app(dict_table(), target="label")
        self.assertEqual(profile["target"]["suggestion"]["status"], "not available")
        self.assertEqual(profile["classes"]["suggested_positive"]["label"], "fraud")  # the rarest label, flagged below
        self.assertEqual(profile["classes"]["suggested_positive"]["status"], "suggested, needs confirmation")


# ---------------------------------------------------------------------------------------------- saved predictions
class EvaluationAdapter(unittest.TestCase):
    def test_wrapper_returns_the_package_report_unchanged(self):
        y, scores = fixed_predictions()
        labels = np.where(y == 1, "pos", "neg").tolist()
        kwargs = dict(positive_label="pos", negative_label="neg", score_kind="probability", threshold=0.4,
                      review_budgets=(6, 12))
        wrapped = evaluate_saved_predictions(labels, {"m": scores.tolist()}, **kwargs)
        direct = evaluate_predictions(labels, {"m": scores.tolist()}, **kwargs)
        for report in (wrapped, direct):
            report.pop("created_utc")
        self.assertEqual(wrapped, direct)
        strict_json(self, wrapped)

    def test_cost_mapping_is_converted_and_checked(self):
        y, scores = fixed_predictions()
        labels = np.where(y == 1, "pos", "neg").tolist()
        costs = {"false_positive_cost": 1, "false_negative_cost": 10, "review_cost": 0.5}
        report = evaluate_saved_predictions(labels, {"m": scores.tolist()}, positive_label="pos", negative_label="neg",
                                            score_kind="probability", costs=costs)
        direct = evaluate_predictions(labels, {"m": scores.tolist()}, positive_label="pos", negative_label="neg",
                                      score_kind="probability", costs=CostAssumptions(1, 10, 0.5))
        self.assertEqual(report["models"]["m"]["threshold_result"]["assumed_total_cost"],
                         direct["models"]["m"]["threshold_result"]["assumed_total_cost"])
        with self.assertRaises(ServiceError) as caught:
            evaluate_saved_predictions(labels, {"m": scores.tolist()}, positive_label="pos", negative_label="neg",
                                       score_kind="probability", costs={"false_positive_cost": 1})
        self.assertEqual(caught.exception.code, "APP_INPUT_INVALID")

    def test_html_is_standalone(self):
        y, scores = fixed_predictions()
        labels = np.where(y == 1, "pos", "neg").tolist()
        report = evaluate_saved_predictions(labels, {"m": scores.tolist()}, positive_label="pos", negative_label="neg",
                                            score_kind="probability")
        page = render_report_html(report)
        self.assertNotIn("<script", page.lower())
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)


# ---------------------------------------------------------------------------------------------- (a) metric parity
@needs_legacy
class MetricParity(unittest.TestCase):
    """Legacy ``engine._metrics`` versus ``evaluate_predictions`` on identical predictions and thresholds."""

    THRESHOLDS = (0.3, 0.5, 0.7)

    def legacy(self, y, scores, threshold):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return legacy._metrics(y, (scores >= threshold).astype(int), scores)

    def test_definitions_that_match_give_equal_values(self):
        y, scores = fixed_predictions()
        for threshold in self.THRESHOLDS:
            with self.subTest(threshold=threshold):
                old = self.legacy(y, scores, threshold)
                new = package_evaluation(y, scores, threshold)
                result, metrics = new["threshold_result"], new["metrics"]
                pairs = {"recall": result["recall"], "precision": result["precision"], "f1": result["f1"],
                         "balanced_accuracy": result["balanced_accuracy"], "accuracy": result["accuracy"],
                         "roc_auc": metrics["roc_auc"], "pr_auc": metrics["average_precision"]}
                for key, value in pairs.items():
                    self.assertEqual(old[key], round(value, 4), key)

    def test_threshold_metrics_equal_the_scikit_learn_definitions_before_rounding(self):
        from sklearn import metrics
        y, scores = fixed_predictions()
        for threshold in self.THRESHOLDS:
            predicted = (scores >= threshold).astype(int)
            result = package_evaluation(y, scores, threshold)["threshold_result"]
            self.assertAlmostEqual(result["recall"], metrics.recall_score(y, predicted), places=12)
            self.assertAlmostEqual(result["precision"], metrics.precision_score(y, predicted), places=12)
            self.assertAlmostEqual(result["f1"], metrics.f1_score(y, predicted), places=12)
            self.assertAlmostEqual(result["balanced_accuracy"], metrics.balanced_accuracy_score(y, predicted), places=12)
            self.assertAlmostEqual(result["accuracy"], metrics.accuracy_score(y, predicted), places=12)

    def test_average_precision_is_the_same_value_under_a_different_name_and_is_not_trapezoidal_area(self):
        from sklearn import metrics
        y, scores = fixed_predictions()
        new = package_evaluation(y, scores, 0.5)["metrics"]
        old = self.legacy(y, scores, 0.5)
        self.assertEqual(old["pr_auc"], round(new["average_precision"], 4))  # both call sklearn's average_precision_score
        self.assertAlmostEqual(new["average_precision"], metrics.average_precision_score(y, scores), places=12)
        precision, recall, _ = metrics.precision_recall_curve(y, scores)
        trapezoid = metrics.auc(recall, precision)
        # The legacy key "pr_auc" is therefore average precision. Trapezoidal PR area is a different number.
        self.assertGreater(abs(trapezoid - new["average_precision"]), 1e-3)

    def test_gmean_exists_only_in_the_legacy_output_and_equals_sqrt_of_recall_times_specificity(self):
        y, scores = fixed_predictions()
        for threshold in self.THRESHOLDS:
            old = self.legacy(y, scores, threshold)
            result = package_evaluation(y, scores, threshold)["threshold_result"]
            self.assertNotIn("gmean", result)
            self.assertEqual(old["gmean"], round(math.sqrt(result["recall"] * result["specificity"]), 4))

    def test_legacy_rounds_to_four_decimals_and_the_package_does_not(self):
        y, scores = fixed_predictions()
        old = self.legacy(y, scores, 0.5)
        new = package_evaluation(y, scores, 0.5)
        values = [new["threshold_result"][key] for key in ("recall", "precision", "f1", "balanced_accuracy")]
        values += [new["metrics"]["roc_auc"], new["metrics"]["average_precision"]]
        self.assertTrue(all(round(value, 4) == value for value in old.values()))
        self.assertTrue(any(round(value, 4) != value for value in values), "fixture should expose the rounding")

    def test_undefined_metrics_are_null_in_the_package_and_zero_or_one_in_the_legacy_engine(self):
        y, scores = fixed_predictions()
        # No row flagged: precision has no denominator.
        old = self.legacy(y, scores, 1.0)
        new = package_evaluation(y, scores, 1.0)["threshold_result"]
        self.assertEqual(old["precision"], 0.0)  # zero_division=0
        self.assertIsNone(new["precision"])
        # One class only: legacy reports perfect recall and precision, the package reports unavailable.
        only_negative = np.zeros(20, dtype=int)
        flat = np.round(np.linspace(0.05, 0.4, 20), 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            single = legacy._metrics(only_negative, np.zeros(20, dtype=int), flat)
        report = package_evaluation(only_negative, flat, 0.5)
        self.assertEqual((single["recall"], single["precision"]), (1.0, 1.0))
        self.assertNotIn("roc_auc", single)
        self.assertIsNone(report["threshold_result"]["recall"])
        self.assertIsNone(report["metrics"]["roc_auc"])
        self.assertIsNone(report["metrics"]["average_precision"])
        strict_json(self, report)

    def test_threshold_source_fixed_rule_versus_validation_selection(self):
        # Same decision rule: score >= threshold, and 0.5 is the package default for probability scores.
        self.assertEqual(legacy._ProbaModel([0.49, 0.5, 0.51]).predict(None).tolist(), [0, 1, 1])
        y, scores = fixed_predictions()
        labels = np.where(y == 1, "pos", "neg").tolist()
        default = evaluate_predictions(labels, {"m": scores.tolist()}, positive_label="pos", negative_label="neg",
                                       score_kind="probability")
        self.assertEqual(default["policies"]["threshold"], 0.5)
        self.assertEqual(default["policies"]["threshold_rule"], "score >= threshold")
        # The training path does not use 0.5: it selects on validation data only.
        validation_y = np.array([1] * 6 + [0] * 24)
        validation_scores = np.array([0.2, 0.25, 0.3, 0.35, 0.3, 0.22] + [0.004 * i for i in range(24)])
        selection = select_threshold(validation_y, validation_scores)
        self.assertEqual((selection["source"], selection["test_labels_used"]), ("validation_only", False))
        self.assertLess(selection["threshold"], 0.5)
        self.assertEqual(selection["validation_objective_value"], 1.0)
        self.assertEqual(self.legacy(validation_y, validation_scores, 0.5)["recall"], 0.0)  # fixed 0.5 would flag nothing


# ---------------------------------------------------------------------------------------------- recipe map
class RecipeMap(unittest.TestCase):
    def test_entries_are_complete_and_point_at_real_registry_ids(self):
        entries = legacy_recipe_map()
        self.assertEqual(len({(item["kind"], item["legacy_name"]) for item in entries}), len(entries))
        for item in entries:
            with self.subTest(name=item["legacy_name"]):
                self.assertIn(item["disposition"], ("recipe", "legacy only"))
                self.assertTrue(item["reason"])
                if item["disposition"] == "recipe":
                    self.assertIn(item["recipe_id"], REGISTRY)
                    self.assertIn(item["mixed_recipe_id"], MIXED_REGISTRY)
                else:
                    self.assertIsNone(item["recipe_id"])
                if item["nearest_recipe"]:
                    self.assertIn(item["nearest_recipe"], REGISTRY)
        strict_json(self, entries)

    def test_plain_is_not_mapped_to_the_unweighted_research_forest(self):
        plain = services.recipe_for_legacy("plain", "model")
        self.assertEqual(plain["disposition"], "legacy only")
        self.assertEqual(REGISTRY["legacy_rf_weighted_200"].scope, "legacy_app")
        self.assertEqual(plain["nearest_recipe"], "rf_weighted")
        self.assertNotEqual(plain["nearest_recipe"], "rf_raw")
        self.assertEqual(services.recipe_for_legacy("random_forest", "classifier")["recipe_id"], "rf_raw")

    @needs_legacy
    def test_every_name_used_by_the_live_engine_is_mapped(self):
        mapped = {(item["kind"], item["legacy_name"]) for item in legacy_recipe_map()}
        for name in legacy.METHODS:
            self.assertIn(("resampling_method", name), mapped)
        for name in legacy.CLASSIFIERS:
            self.assertIn(("classifier", name), mapped)
        for name in ("plain", "csrboost", "erf"):  # the three models of classify_compare
            self.assertIn(("model", name), mapped)

    @needs_legacy
    def test_mapped_recipes_agree_with_the_legacy_constants_they_claim_to_share(self):
        from models import make_plain
        from models.erf_csrboost import N_BAGS, N_TREES
        erf = REGISTRY["erf"].to_record()["params"]["estimator"]
        self.assertEqual((N_BAGS, N_TREES), (erf["n_bags"], erf["trees_per_bag"]))
        plain = make_plain(weighted=True)
        self.assertEqual((plain.n_estimators, plain.class_weight), (200, "balanced"))
        spec = REGISTRY["legacy_rf_weighted_200"].to_record()["params"]["estimator"]
        self.assertEqual((spec["n_estimators"], spec["class_weight"]), (200, "balanced"))


# ---------------------------------------------------------------------------------------------- (b) end to end
@needs_legacy
class EndToEndMixedFixture(unittest.TestCase):
    """One mixed-type table through the legacy engine and through the service (real worker subprocesses)."""

    @classmethod
    def setUpClass(cls):
        cls.csv = mixed_fixture_csv()
        cls.legacy_result = legacy.analyze(cls.csv, "target", "clustered_smote", do_classify=True, classifier="erf", folds=1)
        cls.profile = profile_for_app(cls.csv.encode("utf-8"), "fixture.csv")
        cls.models = ("rf_raw", "rf_weighted", "csrboost", "erf")
        cls.result = train_for_app(cls.csv.encode("utf-8"), filename="fixture.csv", models=cls.models, timeout_seconds=300,
                                   **confirmed_request(cls.profile))

    def test_both_paths_succeed(self):
        self.assertTrue(self.legacy_result["ok"])
        for name in ("plain", "csrboost", "erf"):
            self.assertIsNotNone(self.legacy_result["classify"][name], name)
        for name in self.models:
            self.assertEqual(self.result["outcomes"][name]["status"], "success", self.result["outcomes"][name].get("error"))
        self.assertEqual(self.result["schema_version"], "training-1.1")
        self.assertEqual(self.result["preprocessing"]["adapter"]["recipe_variant"], "mixed")
        strict_json(self, self.result)
        strict_json(self, self.profile)

    def test_class_counts_agree(self):
        old, new = self.legacy_result["full_counts"], self.result["label_mapping"]
        self.assertEqual(self.legacy_result["positive_label"], "1")
        self.assertEqual(old["1"], new["counts"]["positive_rows"])
        self.assertEqual(old["0"], new["counts"]["negative_rows"])
        self.assertEqual(self.legacy_result["n_rows"], new["counts"]["effective_rows"])
        self.assertEqual({item["label"]: item["count"] for item in new["original_labels"]}, old)
        self.assertEqual(self.profile["classes"]["labels"], [{"label": "0", "count": old["0"]}, {"label": "1", "count": old["1"]}])
        parts = self.result["split"]["achieved"]["partitions"]
        self.assertEqual(sum(item["rows"] for item in parts.values()), self.legacy_result["n_rows"])
        self.assertEqual(sum(item["positive_rows"] for item in parts.values()), old["1"])

    def test_split_recipe_single_quarter_versus_sixty_twenty_twenty(self):
        rows = self.legacy_result["n_rows"]
        self.assertEqual(legacy.TEST_SIZE, 0.25)
        self.assertEqual(sum(self.legacy_result["test_counts"].values()), math.ceil(0.25 * rows))
        self.assertEqual(sum(self.legacy_result["before_train"].values()) + sum(self.legacy_result["test_counts"].values()), rows)
        parts = self.result["split"]["achieved"]["partitions"]
        self.assertEqual(set(parts), {"train", "validation", "test"})
        for name, target in (("train", 0.6), ("validation", 0.2), ("test", 0.2)):
            self.assertAlmostEqual(parts[name]["fraction_of_rows"], target, delta=0.02, msg=name)
        self.assertFalse(self.result["split"]["achieved"]["fallback_used"])
        self.assertTrue(self.result["split"]["achieved"]["identical_across_models"])

    def test_threshold_fixed_half_versus_validation_selected(self):
        self.assertEqual(legacy._ProbaModel([0.499, 0.5]).predict(None).tolist(), [0, 1])
        self.assertFalse(any("threshold" in key for key in self.legacy_result["classify"]["erf"]))
        for name in self.models:
            outcome = self.result["outcomes"][name]
            selection = outcome["execution"]["threshold_selection"]
            self.assertEqual(selection["source"], "validation_only", name)
            self.assertFalse(selection["test_labels_used"])
            self.assertIn(selection["threshold"], [k / 100 for k in range(101)])
            self.assertEqual(outcome["report"]["models"][name]["threshold_result"]["threshold"], selection["threshold"])

    def test_plain_forest_naming_and_settings(self):
        from models import make_plain
        legacy_plain = make_plain(weighted=True)
        self.assertEqual((legacy_plain.n_estimators, legacy_plain.class_weight), (200, "balanced"))
        raw = self.result["model_records"]["rf_raw"]
        self.assertEqual(raw["recipe_id"], "rare_event_eval/rf_raw-mixed/1")
        self.assertEqual((raw["params"]["estimator"]["n_estimators"], raw["params"]["estimator"]["class_weight"]), (100, None))
        weighted = self.result["model_records"]["rf_weighted"]["params"]["estimator"]
        self.assertEqual((weighted["n_estimators"], weighted["class_weight"]), (100, "balanced"))
        # The weighted forest is never called plain; only the unweighted forest carries that word.
        self.assertNotIn("plain", self.result["model_records"]["rf_weighted"]["display_name"].lower())
        self.assertIn("unweighted", raw["display_name"].lower())

    def test_erf_capacity_is_the_same_and_it_is_not_boosting(self):
        erf = self.result["model_records"]["erf"]["params"]["estimator"]
        self.assertEqual((legacy.N_BAGS, legacy.N_TREES), (erf["n_bags"], erf["trees_per_bag"]))
        self.assertEqual(erf["class"], "MixedERFClassifier")
        self.assertNotIn("boost", json.dumps(erf).lower())

    def test_csrboost_recipe_uses_categorical_aware_resampling_instead_of_one_hot_smote(self):
        record = self.result["model_records"]["csrboost"]
        self.assertEqual(record["recipe_id"], "rare_event_eval/csrboost-mixed/1")
        self.assertEqual(record["params"]["resampling"]["class"], "SMOTENC")
        self.assertIn("SMOTENC", self.result["preprocessing"]["adapter"]["oversampler"])
        events = self.result["outcomes"]["csrboost"]["execution"]["resampling_events"]
        self.assertTrue(events)
        self.assertEqual(self.result["fallback_policy"]["policy"], "strict")
        self.assertFalse(self.result["fallback_policy"]["fallback_used"])

    def test_training_cap_difference(self):
        from rare_event_eval.tabular import TabularLimits
        self.assertEqual(legacy.CAP_COMPARE, 4000)
        self.assertEqual(TabularLimits().max_rows, 20000)  # the package refuses a larger table instead of sampling it

    def test_resampler_fallback_behavior(self):
        # Legacy: a SMOTE that cannot run is replaced by random oversampling, with only a text note.
        frame = legacy.pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.linspace(1, 0, 20)})
        y = np.zeros(20, dtype=int)
        y[0] = 1
        resampled, labels, note = legacy.resample_train("smote", frame, y, ["a", "b"], [])
        self.assertIn("fell back to random oversampling", note)
        self.assertEqual(np.bincount(labels).tolist(), [19, 19])
        # Legacy CSRBoost helper: silent return of the unresampled data, no note at all.
        from models.csrboost_exact import _fit_resample_safe

        class Broken:
            def fit_resample(self, X, y):
                raise RuntimeError("cannot resample")
        X = np.arange(40, dtype=float).reshape(20, 2)
        Xo, yo = _fit_resample_safe(Broken(), X, np.array([1] * 5 + [0] * 15))
        self.assertIs(Xo, X)
        # Service: the same situation is a failed model with a reason code and no substitute recipe.
        table = dict_table(n=60, extra={"when": [(date(2026, 1, 1) + timedelta(days=i // 3)).isoformat() for i in range(60)]})
        table["label"] = ["fraud" if i in (10, 40, 55) else "ok" for i in range(60)]
        runner = InProcessRunner()
        with mock.patch("rare_event_eval.mixed.run_worker_directory", runner):
            result = train_for_app(table, target="label", positive_label="fraud", negative_label="ok",
                                   split_mode="chronological", time_column="when",
                                   models=("logistic", "smote_rf", "csrboost"))
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        for name in ("smote_rf", "csrboost"):
            self.assertEqual((result["outcomes"][name]["status"], result["outcomes"][name]["reason_code"]),
                             ("failed", "RESAMPLING_INFEASIBLE"), name)
            self.assertIn("no fallback", result["outcomes"][name]["error"])
        self.assertFalse(result["fallback_policy"]["fallback_used"])
        self.assertEqual({item["model"] for item in result["failures"]}, {"smote_rf", "csrboost"})


# ---------------------------------------------------------------------------------------------- (c) regression
@needs_legacy_readers
class MixedInputRegression(unittest.TestCase):
    """Every table-shaped input of the legacy torture test profiles and trains through the service.

    The model comparison runs the real worker code in this process (one logistic model per input) to keep the
    run short; the real subprocess path is covered by the end-to-end and sample tests.
    """

    # Intentional differences, each asserted below: the legacy app did these without asking.
    CONFIRMATION_REQUIRED = {
        "quoted_commas": "identifier-like text column (200 distinct values) must be excluded explicitly",
        "dates_money_percent_inf": "date strings are read as text and are identifier-like; must be excluded explicitly",
    }
    MISSING_TARGET_POLICY = {"ragged_missing_fields": "a row with a missing target is rejected unless drop_reported is chosen"}

    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.refused = torture_inputs()

    def train(self, raw, name, **extra):
        profile = profile_for_app(raw, name)
        request = confirmed_request(profile)
        request.update(extra)
        runner = InProcessRunner()
        with mock.patch("rare_event_eval.mixed.run_worker_directory", runner):
            return profile, train_for_app(raw, filename=name, models=("logistic",), **request)

    def test_inputs_the_legacy_test_accepts_profile_and_train(self):
        accepted = [key for key in self.inputs if key not in self.refused]
        self.assertGreaterEqual(len(accepted), 36)
        for key in accepted:
            raw, name = self.inputs[key]
            with self.subTest(case=key):
                extra = {"missing_policy": "drop_reported"} if key in self.MISSING_TARGET_POLICY else {}
                profile, result = self.train(raw, name, **extra)
                self.assertEqual(result["outcomes"]["logistic"]["status"], "success", result["outcomes"]["logistic"].get("error"))
                after = profile["limits"]["status_if_suggested_exclusions_applied"]
                self.assertTrue(after is None or after["status"] == "ok")
                self.assertEqual(result["input_parsing"]["details"]["input_sha256"], hashlib.sha256(raw).hexdigest())
                strict_json(self, result)

    def test_inputs_the_legacy_test_refuses_are_refused_with_a_typed_error(self):
        self.assertEqual(self.refused, {"single_column", "empty_file", "garbage", "data.parquet", "data.sav"})
        for key in sorted(self.refused):
            raw, name = self.inputs[key]
            with self.subTest(case=key):
                with self.assertRaises(ServiceError) as caught:
                    self.train(raw, name)
                self.assertTrue(caught.exception.code)
                self.assertNotIn("Traceback", str(caught.exception))

    def test_intentional_difference_identifier_like_columns_need_explicit_exclusion(self):
        for key in self.CONFIRMATION_REQUIRED:
            raw, name = self.inputs[key]
            with self.subTest(case=key):
                profile = profile_for_app(raw, name)
                self.assertEqual(profile["limits"]["status"], "exceeded")
                self.assertEqual(profile["limits"]["violations"][0]["code"], "TABLE_CARDINALITY_EXCEEDED")
                self.assertTrue(profile["suggested_exclusions"])
                self.assertEqual(profile["limits"]["status_if_suggested_exclusions_applied"]["status"], "ok")
                request = confirmed_request(profile)
                request["exclude_columns"] = []
                with self.assertRaises(ServiceError) as caught:
                    train_for_app(raw, filename=name, models=("logistic",), **request)
                self.assertEqual(caught.exception.code, "TABLE_CARDINALITY_EXCEEDED")

    def test_intentional_difference_missing_targets_are_never_dropped_silently(self):
        raw, name = self.inputs["ragged_missing_fields"]
        profile = profile_for_app(raw, name)
        self.assertEqual(profile["classes"]["missing_target_rows"], 1)
        with self.assertRaises(ServiceError) as caught:
            train_for_app(raw, filename=name, models=("logistic",), **confirmed_request(profile))
        self.assertEqual(caught.exception.code, "LABEL_MISSING_TARGET")
        _, result = self.train(raw, name, missing_policy="drop_reported")
        self.assertEqual(result["input_parsing"]["rows_excluded"], 1)
        self.assertEqual(len(result["input_parsing"]["row_exclusions"]), 1)

    def test_legacy_reader_warnings_are_carried_into_the_report(self):
        raw, name = self.inputs["ragged_extra_fields"]
        _, result = self.train(raw, name)
        details = result["input_parsing"]["details"]
        self.assertEqual(details["parser"], "legacy_engine.read_csv")
        self.assertEqual(details["malformed_rows_skipped_by_reader"], 2)
        self.assertTrue(any("malformed" in item for item in details["read_warnings"]))

    def test_bundled_samples_profile_and_train_with_real_workers(self):
        for path in sorted(SAMPLES.glob("*.csv")):
            with self.subTest(sample=path.name):
                raw = path.read_bytes()
                profile = profile_for_app(raw, path.name)
                self.assertEqual(profile["target"]["column"], "target")
                self.assertEqual(profile["classes"]["suggested_positive"]["label"], "1")
                self.assertEqual(profile["limits"]["status"], "ok")
                result = train_for_app(raw, filename=path.name, models=("logistic",), timeout_seconds=300,
                                       **confirmed_request(profile))
                self.assertEqual(result["outcomes"]["logistic"]["status"], "success", result["outcomes"]["logistic"].get("error"))
                counts = {item["label"]: item["count"] for item in profile["classes"]["labels"]}
                self.assertEqual(result["label_mapping"]["counts"]["positive_rows"], counts["1"])


# ---------------------------------------------------------------------------------------------- (d) leakage
class LeakageBoundaryThroughTheService(unittest.TestCase):
    """Spy on every train-fitted transform and the final fits while the service trains in this process."""

    def run_service(self, data, models, filename=""):
        runner = InProcessRunner()
        with mock.patch("rare_event_eval.mixed.run_worker_directory", runner):
            return train_for_app(data, filename=filename, target="label", positive_label="fraud", negative_label="ok",
                                 models=models)

    def spied_run(self, data, models, filename=""):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTENC
        from rare_event_eval.mixed_models import TabularPreprocessor

        seen = {"preprocessor": [], "imputer": [], "scaler": [], "logistic": [], "forest": [], "smotenc": []}

        def spy(cls, key, method="fit", ids=True):
            original = getattr(cls, method)

            def wrapper(self, X, *args, **kwargs):
                X = np.asarray(X)
                seen[key].append(X[:, 0].astype(float).astype(int).tolist() if ids else len(X))
                return original(self, X, *args, **kwargs)
            return mock.patch.object(cls, method, wrapper)

        with spy(TabularPreprocessor, "preprocessor"), spy(SimpleImputer, "imputer"), spy(StandardScaler, "scaler"), \
                spy(LogisticRegression, "logistic", ids=False), spy(RandomForestClassifier, "forest", ids=False), \
                spy(SMOTENC, "smotenc", "fit_resample", ids=False):
            result = self.run_service(data, models, filename)
        return result, seen

    def check(self, result, seen, columns):
        y = np.array([label == "fraud" for label in columns["label"]], dtype=int)
        split = make_split(y, mode="stratified", seed=42)
        train_ids = {1000 + 7 * int(i) for i in split.train}
        later_ids = {1000 + 7 * int(i) for i in np.concatenate([split.validation, split.test])}
        for name in ("logistic", "smote_rf"):
            self.assertEqual(result["outcomes"][name]["status"], "success", result["outcomes"][name].get("error"))
        for key in ("preprocessor", "imputer", "scaler"):
            self.assertGreaterEqual(len(seen[key]), 2, key)  # the standalone record and the model's own fit
            for ids in seen[key]:
                self.assertEqual(set(ids), train_ids, key)
                self.assertFalse(set(ids) & later_ids, key)
        self.assertEqual(seen["logistic"], [len(split.train)])
        self.assertEqual(seen["forest"], [len(split.train) + int((y[split.train] == 0).sum()) - int(y[split.train].sum())])
        self.assertEqual(seen["smotenc"], [len(split.train)])  # SMOTENC sees the training partition only
        self.assertEqual(result["split"]["achieved"]["partitions"]["train"]["rows"], len(split.train))

    def test_table_object_input(self):
        columns = dict_table(row_ids=True)
        result, seen = self.spied_run(columns, ("logistic", "smote_rf"))
        self.check(result, seen, columns)

    @needs_legacy
    def test_upload_bytes_through_the_legacy_readers(self):
        columns = dict_table(row_ids=True)
        result, seen = self.spied_run(to_csv_bytes(columns), ("logistic", "smote_rf"), "table.csv")
        self.check(result, seen, columns)

    def test_changing_validation_and_test_cells_does_not_change_training_statistics(self):
        columns = dict_table(row_ids=True)
        y = np.array([label == "fraud" for label in columns["label"]], dtype=int)
        split = make_split(y, mode="stratified", seed=42)
        changed = {key: list(values) for key, values in columns.items()}
        for index in np.concatenate([split.validation, split.test]):
            changed["income"][index] = "100000"
            changed["city"][index] = "zzz"
        first = self.run_service(columns, ("logistic",))
        second = self.run_service(changed, ("logistic",))
        for key in ("expanded_feature_names", "numeric_columns"):
            self.assertEqual(first["preprocessing"]["record"][key], second["preprocessing"]["record"][key])
        self.assertNotIn("city=zzz", second["preprocessing"]["record"]["expanded_feature_names"])


# ---------------------------------------------------------------------------------------------- (e) report safety
class ReportSafety(unittest.TestCase):
    SCRIPT = "<script>alert(1)</script>"
    IMAGE = "<img src=x onerror=alert(1)>"

    def run_service(self, data, **kwargs):
        runner = InProcessRunner()
        with mock.patch("rare_event_eval.mixed.run_worker_directory", runner):
            return train_for_app(data, **kwargs)

    def test_training_report_is_strict_json_and_html_escapes_labels_and_names(self):
        positive, negative = self.SCRIPT, 'a&b "c"'
        columns = dict_table(labels=(positive, negative), extra={self.IMAGE: ["u" if i % 2 else "v" for i in range(140)]})
        result = self.run_service(columns, target="label", positive_label=positive, negative_label=negative,
                                  models=("logistic", "rf_raw"), description="Fixture " + self.SCRIPT)
        for name in ("logistic", "rf_raw"):
            self.assertEqual(result["outcomes"][name]["status"], "success")
        strict_json(self, result)
        page = render_report_html(result)
        self.assertNotIn(self.SCRIPT, page)
        self.assertNotIn(self.IMAGE, page)
        self.assertNotIn("<script", page.lower())
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", page)
        self.assertIn("a&amp;b", page)

    def test_failed_model_report_is_strict_json_and_visible_in_html(self):
        table = dict_table(n=60, extra={"when": [(date(2026, 1, 1) + timedelta(days=i // 3)).isoformat() for i in range(60)]})
        table["label"] = ["fraud" if i in (10, 40, 55) else "ok" for i in range(60)]
        result = self.run_service(table, target="label", positive_label="fraud", negative_label="ok", split_mode="chronological",
                                  time_column="when", models=("logistic", "smote_rf"))
        strict_json(self, result)
        page = render_report_html(result)
        self.assertIn("smote_rf", page)
        self.assertIn("were not silently removed", page)

    def test_undefined_metrics_are_null_not_nan_in_the_evaluation_report(self):
        labels = ["neg"] * 20
        report = evaluate_saved_predictions(labels, {"m": [round(0.02 * i, 3) for i in range(20)]}, positive_label="pos",
                                            negative_label="neg", score_kind="probability", threshold=0.3)
        model = report["models"]["m"]
        self.assertIsNone(model["metrics"]["average_precision"])
        self.assertIsNone(model["metrics"]["roc_auc"])
        self.assertIsNone(model["threshold_result"]["recall"])
        strict_json(self, report)

    def test_evaluation_html_escapes_labels(self):
        labels = [self.SCRIPT] * 5 + ['x"y'] * 15
        scores = [0.9, 0.8, 0.7, 0.4, 0.3] + [round(0.02 * i, 3) for i in range(15)]
        report = evaluate_saved_predictions(labels, {"<b>model</b>": scores}, positive_label=self.SCRIPT,
                                            negative_label='x"y', score_kind="probability")
        page = render_report_html(report)
        self.assertNotIn(self.SCRIPT, page)
        self.assertNotIn("<b>model</b>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("&lt;b&gt;model&lt;/b&gt;", page)

    def test_no_en_or_em_dashes_in_service_sources_or_messages(self):
        banned = (chr(0x2013), chr(0x2014))
        for path in list((BACKEND / "services").glob("*.py")) + [Path(__file__)]:
            text = path.read_text(encoding="utf-8")
            self.assertFalse(any(char in text for char in banned), path.name)
        strict_json(self, legacy_recipe_map())
        self.assertFalse(any(char in json.dumps(legacy_recipe_map(), ensure_ascii=False) for char in banned))


# ---------------------------------------------------------------------------------------------- beside the live server
class WithAndWithoutTheLiveServer(unittest.TestCase):
    def test_importing_the_service_does_not_import_the_server_or_the_engine(self):
        code = ("import sys; sys.path.insert(0, %r); import services; "
                "print(int('server' in sys.modules), int('engine' in sys.modules), int('track' in sys.modules))" % str(BACKEND))
        completed = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=120,
                                   env=dict(os.environ))
        self.assertEqual(completed.returncode, 0, completed.stderr[-400:])
        self.assertEqual(completed.stdout.split(), ["0", "0", "0"])

    def test_service_works_without_a_server_process(self):
        columns = dict_table()
        profile = profile_for_app(columns, target="label")
        strict_json(self, profile)
        self.assertEqual(profile["classes"]["distinct_labels"], 2)
        report = evaluate_saved_predictions(["a", "b", "a", "b"], {"m": [0.9, 0.1, 0.8, 0.2]}, positive_label="a",
                                            negative_label="b", score_kind="probability")
        self.assertEqual(report["models"]["m"]["metrics"]["roc_auc"], 1.0)

    @needs_legacy
    def test_service_profile_agrees_with_the_running_live_server(self):
        from http.server import ThreadingHTTPServer
        import server
        csv = mixed_fixture_csv()
        httpd = None
        for _ in range(5):  # some Windows setups refuse an ephemeral port now and then (WinError 10013)
            try:
                httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
                break
            except OSError:
                continue
        if httpd is None:
            self.skipTest("this environment refuses to bind a local socket")
        original = server.track.log_event
        server.track.log_event = lambda *args, **kwargs: None
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
            connection.request("POST", "/api/profile", body=json.dumps({"csv": csv}), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            live = json.loads(response.read())
            connection.close()
            self.assertEqual(response.status, 200)
            served = profile_for_app(csv.encode("utf-8"), "fixture.csv")  # the service answers while the server is up
        finally:
            httpd.shutdown()
            httpd.server_close()
            server.track.log_event = original
        self.assertEqual(served["target"]["suggestion"]["column"], live["suggested_target"])
        self.assertEqual(served["rows"], live["n_rows"])
        self.assertEqual([item["column"] for item in served["schema_columns"]], [item["column"] for item in live["columns"]])
        self.assertEqual(served["target"]["suggestion"]["status"], "suggested, needs confirmation")


if __name__ == "__main__":
    unittest.main()
