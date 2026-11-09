from __future__ import annotations

from itertools import combinations
import json
import math
import unittest

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from rare_event_eval import CostAssumptions, evaluate_predictions, render_html


def evaluate(labels, scores, **kwargs):
    return evaluate_predictions(labels, {"model": scores}, positive_label="event",
                                negative_label="clear", score_kind=kwargs.pop("score_kind", "probability"),
                                **kwargs)


class MetricsAndPolicies(unittest.TestCase):
    def test_known_example_has_hand_calculated_results(self):
        report = evaluate(["clear", "clear", "event", "event"], [.1, .4, .35, .8],
                          review_budgets=[2], costs=CostAssumptions(2, 10, 1))
        result = report["models"]["model"]
        self.assertAlmostEqual(result["metrics"]["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["metrics"]["roc_auc"], .75)
        self.assertAlmostEqual(result["metrics"]["brier_score"], .158125)
        fixed = result["threshold_result"]
        self.assertEqual([fixed[key] for key in ("true_positives", "false_positives", "false_negatives", "true_negatives")],
                         [1, 0, 1, 2])
        self.assertEqual(fixed["assumed_total_cost"], 11)
        self.assertEqual(result["review_budget_results"][0]["assumed_total_cost"], 14)

    def test_reference_metrics_with_ties_and_exact_endpoints(self):
        rng = np.random.default_rng(23)
        for size in (4, 31, 200):
            with self.subTest(size=size):
                y = rng.integers(0, 2, size=size)
                y[:2] = [0, 1]
                scores = rng.choice([0., .2, .5, .8, 1.], size=size)
                labels = ["event" if value else "clear" for value in y]
                metrics = evaluate(labels, scores)["models"]["model"]["metrics"]
                self.assertAlmostEqual(metrics["average_precision"], average_precision_score(y, scores))
                self.assertAlmostEqual(metrics["roc_auc"], roc_auc_score(y, scores))
                self.assertAlmostEqual(metrics["brier_score"], brier_score_loss(y, scores))
                self.assertAlmostEqual(metrics["log_loss"], log_loss(y, scores))

    def test_majority_positive_and_non_numeric_labels_are_preserved(self):
        labels = ["event"] * 9 + ["clear"]
        report = evaluate(labels, [.9] * 9 + [.1])
        self.assertEqual(report["data"]["positive_rows"], 9)
        self.assertEqual(report["task"]["positive_label"], "event")
        self.assertEqual(report["models"]["model"]["threshold_result"]["recall"], 1)
        self.assertEqual(report["data"]["rows_dropped"], 0)

    def test_changing_positive_class_requires_correct_score_orientation(self):
        labels = ["clear", "clear", "event", "event"]
        scores = np.asarray([.1, .4, .35, .8])
        first = evaluate(labels, scores)
        second = evaluate_predictions(labels, {"model": 1 - scores}, positive_label="clear",
                                      negative_label="event", score_kind="probability")
        a = first["models"]["model"]["threshold_result"]
        b = second["models"]["model"]["threshold_result"]
        self.assertEqual(a["true_positives"], b["true_negatives"])
        self.assertEqual(a["false_negatives"], b["false_positives"])
        self.assertAlmostEqual(first["models"]["model"]["metrics"]["brier_score"],
                               second["models"]["model"]["metrics"]["brier_score"])

    def test_threshold_equality_is_flagged(self):
        result = evaluate(["event", "clear"], [.5, .1])["models"]["model"]["threshold_result"]
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["flagged"], 1)

    def test_top_k_tie_bounds_match_all_possible_allocations(self):
        labels = ["clear", "event", "clear", "event", "event", "clear"]
        scores = [.9, .5, .5, .5, .5, .1]
        result = evaluate(labels, scores, review_budgets=[3])["models"]["model"]["review_budget_results"][0]
        possible = [sum(labels[index] == "event" for index in allocation)
                    for allocation in combinations([1, 2, 3, 4], 2)]
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["boundary_tie"]["true_positives_min"], min(possible))
        self.assertEqual(result["boundary_tie"]["true_positives_max"], max(possible))
        perm = [0, 3, 4, 1, 2, 5]
        reordered = evaluate([labels[i] for i in perm], [scores[i] for i in perm], review_budgets=[3])
        other = reordered["models"]["model"]["review_budget_results"][0]
        self.assertEqual(other["true_positives"], 2)
        self.assertEqual(result["boundary_tie"], other["boundary_tie"])

    def test_zero_and_full_budgets(self):
        results = evaluate(["event", "clear", "clear"], [.8, .5, .2], review_budgets=[0, 3])
        zero, full = results["models"]["model"]["review_budget_results"]
        self.assertIsNone(zero["precision"])
        self.assertEqual(zero["recall"], 0)
        self.assertEqual(full["recall"], 1)
        self.assertEqual(full["flagged"], 3)
        self.assertIsNone(full["boundary_tie"])

    def test_ranking_scores_do_not_get_probability_metrics_or_default_threshold(self):
        result = evaluate(["event", "clear"], [7., -2.], score_kind="ranking", review_budgets=[1])
        model = result["models"]["model"]
        self.assertEqual(model["metrics"]["average_precision"], 1)
        self.assertIsNone(model["metrics"]["brier_score"])
        self.assertIsNone(model["metrics"]["log_loss"])
        self.assertIsNone(model["calibration"])
        self.assertIsNone(model["threshold_result"])

    def test_single_class_is_explicit_and_json_remains_standard(self):
        for label in ("event", "clear"):
            report = evaluate([label, label], [0., 1.], review_budgets=[0])
            self.assertIsNone(report["models"]["model"]["metrics"]["roc_auc"])
            self.assertIsNone(report["models"]["model"]["metrics"]["average_precision"])
            self.assertTrue(report["models"]["model"]["warnings"])
            self.assertNotIn("NaN", json.dumps(report, allow_nan=False))

    def test_calibration_bins_account_for_endpoints_and_empty_bins(self):
        result = evaluate(["clear", "event", "event"], [0., .5, 1.], calibration_bins=2)
        bins = result["models"]["model"]["calibration"]["bins"]
        self.assertEqual([row["count"] for row in bins], [1, 2])
        self.assertEqual(bins[1]["mean_score"], .75)
        empty = evaluate(["clear", "event"], [0., 1.], calibration_bins=4)
        self.assertIsNone(empty["models"]["model"]["calibration"]["bins"][1]["mean_score"])

    def test_multiple_models_share_the_population_without_automatic_winner(self):
        report = evaluate_predictions(["event", "clear"], {"first": [.9, .1], "second": [.1, .9]},
                                      positive_label="event", negative_label="clear", score_kind="probability")
        self.assertEqual(list(report["models"]), ["first", "second"])
        self.assertNotIn("winner", report)
        self.assertEqual(report["checks"]["training_leakage"], "not_checked")
        self.assertFalse(report["task"]["role_verified"])


class RejectedInputs(unittest.TestCase):
    def test_invalid_scores_and_alignment(self):
        for scores in ([float("nan"), .5], [float("inf"), .5], [-.1, .5], [1.1, .5], [.5], [[.2], [.3]], ["x", .5]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                evaluate(["event", "clear"], scores)

    def test_unknown_and_missing_labels_are_not_dropped_or_normalized(self):
        for labels in (["event", "third"], ["event", None], ["event", "Clear"], ["event", "clear "], []):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate(labels, [.9, .1])

    def test_invalid_budgets_thresholds_and_calibration(self):
        for kwargs in ({"review_budgets": [-1]}, {"review_budgets": [3]}, {"review_budgets": [True]},
                       {"review_budgets": [1.5]}, {"review_budgets": [1, 1]}, {"threshold": math.nan},
                       {"threshold": 2}, {"threshold": True}, {"calibration_bins": 1},
                       {"calibration_bins": True}, {"score_kind": "anything"}, {"evaluation_role": "train"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate(["event", "clear"], [.9, .1], **kwargs)

    def test_data_description_is_declared_metadata_and_validated(self):
        report = evaluate(["event", "clear"], [.9, .1], data_description="Fictional scores.")
        self.assertEqual(report["data"]["description"], "Fictional scores.")
        self.assertFalse(report["data"]["description_verified"])
        for description in ("", " " * 2, "x" * 501, 12):
            with self.subTest(description=description), self.assertRaises(ValueError):
                evaluate(["event", "clear"], [.9, .1], data_description=description)

    def test_invalid_costs_and_overflow_are_rejected(self):
        for values in ((-1, 2), (math.inf, 2), (1, math.nan), (True, 2)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CostAssumptions(*values)
        with self.assertRaises(ValueError):
            evaluate(["event", "event", "clear"], [0., 0., 0.], costs=CostAssumptions(0, 1e308))

    def test_task_and_model_names_are_validated(self):
        for kwargs in ({"positive_label": "event", "negative_label": "event"},
                       {"positive_label": "", "negative_label": "clear"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_predictions(["event", "clear"], {"model": [.9, .1]}, score_kind="probability", **kwargs)
        with self.assertRaises(ValueError):
            evaluate_predictions(["event", "clear"], {}, positive_label="event", negative_label="clear", score_kind="probability")


class Reports(unittest.TestCase):
    def test_reports_exclude_raw_rows_and_escape_html(self):
        label = '<script>alert("label")</script>'
        model = '<img src=x onerror="alert(1)">'
        report = evaluate_predictions([label, "clear"], {model: [.8123456789, .123456789]},
                                      positive_label=label, negative_label="clear", score_kind="probability",
                                      data_description='<iframe src="https://example.invalid"></iframe>')
        rendered = render_html(report)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img src=x", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("raw_predictions", report)
        self.assertNotIn("y_true", report)
        self.assertNotIn("scores", report["models"][model])
        self.assertIn("not_checked", rendered)
        self.assertNotIn("<iframe", rendered)
        self.assertNotIn('<script src=', rendered)


if __name__ == "__main__":
    unittest.main()
