import json
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rare_event_eval.core import CostAssumptions
from rare_event_eval.datasets import parse_training_csv
from rare_event_eval.estimators import CSRBoostClassifier, ERFClassifier
from rare_event_eval.policies import paired_ap_interval, select_threshold
from rare_event_eval.splits import feature_groups, make_split, split_diagnostics
from rare_event_eval.training import _fit_one, bounded_model_run


class Estimators(unittest.TestCase):
    def setUp(self):
        self.X, y = make_classification(n_samples=80, n_features=5, n_informative=3,
                                        weights=[.75, .25], random_state=19)
        self.y = np.where(y, "rare", "common")

    def test_clone_fit_probability_and_string_class_order(self):
        for model in (ERFClassifier(n_bags=2, trees_per_bag=5), CSRBoostClassifier(n_estimators=5)):
            with self.subTest(model=type(model).__name__):
                copied = clone(model)
                copied.fit(self.X, self.y)
                self.assertEqual(copied.classes_.tolist(), ["common", "rare"])
                probabilities = copied.predict_proba(self.X)
                self.assertEqual(probabilities.shape, (80, 2))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(hasattr(copied, "n_features_in_"))
                self.assertTrue(copied.resampling_events_)

    def test_pipeline_and_fixed_seed_are_reproducible(self):
        first = Pipeline([("scale", StandardScaler()), ("model", ERFClassifier(n_bags=2, trees_per_bag=5))])
        second = clone(first)
        first.fit(self.X, self.y)
        second.fit(self.X, self.y)
        np.testing.assert_array_equal(first.predict_proba(self.X), second.predict_proba(self.X))

    def test_predict_before_fit_is_not_fitted(self):
        for model in (ERFClassifier(), CSRBoostClassifier()):
            with self.subTest(model=type(model).__name__), self.assertRaises(NotFittedError):
                model.predict(self.X)

    def test_invalid_parameters_features_and_target_rejected(self):
        for model in (ERFClassifier(n_bags=0), ERFClassifier(retention=0), ERFClassifier(n_jobs=0),
                      ERFClassifier(resampling="unknown"), CSRBoostClassifier(n_estimators=0)):
            with self.subTest(model=model), self.assertRaises(ValueError):
                model.fit(self.X, self.y)
        with self.assertRaises(ValueError):
            ERFClassifier(n_bags=1).fit(self.X, np.arange(80) % 3)
        with self.assertRaises(ValueError):
            ERFClassifier(n_bags=1).fit(self.X, ["single"] * 80)

    def test_impossible_smote_fails_instead_of_falling_back(self):
        with self.assertRaisesRegex(ValueError, "no fallback"):
            ERFClassifier(n_bags=1, resampling="smote").fit(self.X, ["rare"] + ["common"] * 79)

    def test_ablation_events_are_explicit(self):
        for method in ("none", "smote", "clustered", "clustered_smote"):
            model = ERFClassifier(n_bags=1, trees_per_bag=3, resampling=method).fit(self.X, self.y)
            self.assertEqual(model.resampling_events_[0]["method"], method)


class SplitsAndPolicies(unittest.TestCase):
    def test_groups_are_disjoint_and_rows_not_deleted(self):
        groups = np.repeat(np.arange(30), 4)
        y = np.tile([0, 0, 0, 1], 30)
        split = make_split(y, mode="group", groups=groups)
        sets = [set(groups[part]) for part in (split.train, split.validation, split.test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(len(part) for part in (split.train, split.validation, split.test)), len(y))

    def test_chronology_keeps_timestamp_ties_together(self):
        times = np.repeat(np.arange(30), 4)
        y = np.tile([0, 1, 0, 0], 30)
        split = make_split(y, mode="chronological", times=times)
        self.assertLess(times[split.train].max(), times[split.validation].min())
        self.assertLess(times[split.validation].max(), times[split.test].min())
        with self.assertRaises(ValueError):
            make_split(y, mode="chronological", times=times.astype(str))
        with self.assertRaises(ValueError):
            make_split(y, mode="chronological", times=times, groups=times)

    def test_feature_patterns_and_conflicting_labels_are_diagnostics(self):
        self.assertEqual(feature_groups([[0., np.nan], [-0., np.nan]])[0],
                         feature_groups([[0., np.nan], [-0., np.nan]])[1])
        X = np.tile([[0., np.nan], [1., 2.]], (20, 1))
        y = np.tile([0, 1, 1, 0], 10)
        groups = feature_groups(X)
        self.assertEqual(len(np.unique(groups)), 2)
        split = make_split(y)
        diagnostics = split_diagnostics(X, y, split)
        self.assertEqual(diagnostics["conflicting_feature_patterns"], 2)
        self.assertGreater(diagnostics["test_rows_matching_training_features"], 0)

    def test_infeasible_group_split_does_not_fall_back(self):
        with self.assertRaises(ValueError):
            make_split(np.tile([0, 1], 10), mode="group", groups=["one"] * 20)

    def test_validation_threshold_uses_only_given_validation_data(self):
        selection = select_threshold([0, 0, 1, 1], [.1, .3, .4, .8])
        self.assertAlmostEqual(selection["threshold"], .4)
        self.assertEqual(selection["source"], "validation_only")
        self.assertFalse(selection["test_labels_used"])
        self.assertAlmostEqual(selection["validation_objective_value"], 1)

    def test_cost_selection_is_not_claimed_as_measured_savings(self):
        selection = select_threshold([0, 0, 1, 1], [.1, .7, .4, .8], costs=CostAssumptions(1, 10))
        self.assertEqual(selection["objective"], "minimize_assumed_cost")
        self.assertLessEqual(selection["threshold"], .4)

    def test_paired_bootstrap_has_identical_draws_and_temporal_refusal(self):
        y = np.tile([0, 1], 20)
        scores = np.tile([.1, .9], 20)
        interval = paired_ap_interval(y, scores, scores, repetitions=30)
        self.assertEqual([interval[key] for key in ("estimate", "lower", "upper")], [0, 0, 0])
        self.assertFalse(paired_ap_interval(y, scores, scores, chronological=True)["available"])


class TrainingWorkflow(unittest.TestCase):
    def setUp(self):
        self.X, y = make_classification(n_samples=120, n_features=5, weights=[.75, .25], random_state=7)
        self.labels = np.where(y, "event", "clear")

    def test_preprocessing_fits_training_partition_only(self):
        split = make_split((self.labels == "event").astype(int))
        base = _fit_one(self.X, self.labels, model_name="logistic", positive_label="event", negative_label="clear")
        changed = self.X.copy()
        changed[split.test] += 10000
        altered = _fit_one(changed, self.labels, model_name="logistic", positive_label="event", negative_label="clear")
        self.assertEqual(base["execution"]["threshold_selection"], altered["execution"]["threshold_selection"])
        self.assertEqual(base["execution"]["split"]["partitions"], altered["execution"]["split"]["partitions"])
        self.assertEqual(base["report"]["checks"]["training_preprocessing"], "fitted_on_training_partitions")

    def test_sigmoid_calibration_is_fitted_inside_training(self):
        result = _fit_one(self.X, self.labels, model_name="logistic", positive_label="event",
                          negative_label="clear", calibration="sigmoid")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["execution"]["calibration"], "sigmoid")
        self.assertFalse(result["execution"]["threshold_selection"]["test_labels_used"])

    def test_subprocess_and_failure_records(self):
        result = bounded_model_run(self.X, self.labels, model_name="logistic", positive_label="event", negative_label="clear", timeout_seconds=30)
        self.assertEqual(result["status"], "success", result)
        self.assertGreater(result["worker_wall_seconds"], 0)
        failure = bounded_model_run(self.X, self.labels, model_name="unknown", positive_label="event", negative_label="clear", timeout_seconds=30)
        self.assertEqual(failure["status"], "error")

    def test_numeric_parser_excludes_target_group_and_time(self):
        raw = b"x,target,entity,when\n1,event,a,2026-01-01\n,clear,b,2026-01-02\n"
        parsed = parse_training_csv(raw, target="target", group_column="entity", time_column="when")
        self.assertEqual(parsed["feature_names"], ["x"])
        self.assertTrue(np.isnan(parsed["X"][1, 0]))
        self.assertLess(parsed["times"][0], parsed["times"][1])
        with self.assertRaises(ValueError):
            parse_training_csv(b"x,target\ncategory,event\n", target="target")


if __name__ == "__main__":
    unittest.main()
