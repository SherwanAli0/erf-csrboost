"""Leakage spies for the fitted steps that earlier spies did not cover.

``test_mixed.py`` already records the rows seen by the preprocessing fits and the final estimators. This module
adds three more fitted steps and proves, from the identity of the rows each call actually received, that they
are fit on training-partition rows only (never validation or test rows):

* ``CalibratedClassifierCV`` (sigmoid), including its inner estimator fits, its held-out inner folds and the
  sigmoid calibrators;
* ``KMeans`` (exact, CSRBoost-style) and ``MiniBatchKMeans`` (ERF) used for clustered majority retention;
* ``SMOTENC`` (mixed tables) and ``SMOTE`` (numeric tables).

How rows are identified. Every table has a ``row_id`` predictor holding the row position, so each row is
unique. After preprocessing, a row is no longer recognizable by its raw value, so the spy rebuilds the
representation of ALL rows (train, validation and test) with the preprocessor that the code under test has just
fitted, and looks each row received by KMeans or SMOTE up by value. A row that cannot be found is reported as
unmatched and fails the test, so a changed representation cannot pass silently.

Mutations these tests catch (each was applied to a scratch copy of the source and made the named tests fail;
see the B4 report):

* ``model.fit(Xtr, ytr)`` replaced by a fit on train plus validation rows (in ``mixed.fit_tabular_model`` or
  ``training._fit_one``): the KMeans and SMOTE row sets then contain validation ids and are larger than the
  training class sets; ``CalibratedClassifierCV.fit`` receives more than the training rows.
* The inner calibration folds built over train plus validation rows instead of the training rows: the held-out
  folds no longer partition the training ids, ``CalibratedClassifierCV.fit`` sees validation ids and the sigmoid
  calibrators are fit on more rows than the training partition holds.
* A step that refits the preprocessor on more rows than the estimator receives: the looked-up rows are
  unmatched.
"""
from contextlib import ExitStack
import unittest
from unittest import mock

import numpy as np
from imblearn.base import BaseSampler
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mixed_support import mixed_table, numeric_table, patched_runner, to_arrays
from rare_event_eval.labels import LabelSpec
from rare_event_eval.mixed import train_comparison_tabular
from rare_event_eval.mixed_models import TabularPreprocessor
from rare_event_eval.splits import make_split
from rare_event_eval.training import _fit_one


def raw_ids(X):
    """Row ids from the first column (the ``row_id`` predictor, untouched by imputation)."""
    return np.asarray(X)[:, 0].astype(float).astype(int).tolist()


def fingerprints(matrix):
    rounded = np.round(np.asarray(matrix, dtype=float), 9) + 0.0
    keys = [row.tobytes() for row in rounded]
    mapping = {key: position for position, key in enumerate(keys)}  # row id equals row position
    assert len(mapping) == len(keys), "rows must be distinguishable"
    return mapping


class Probe:
    """Patches the fitted steps and records which rows each call received, in order."""

    def __init__(self, family, X_all):
        self.family, self.X_all = family, X_all
        self.events, self.sigmoid_sizes = [], []
        self.in_window, self.muted = False, False
        self.current, self.imputer = None, None
        self.stack = ExitStack()

    def __enter__(self):
        patch = lambda cls, name, wrapper: self.stack.enter_context(mock.patch.object(cls, name, wrapper(getattr(cls, name))))
        if self.family == "mixed":
            patch(TabularPreprocessor, "fit", self._mixed_fit)
            patch(TabularPreprocessor, "transform", self._transform)
        else:
            patch(SimpleImputer, "fit", self._imputer_fit)
            patch(StandardScaler, "fit", self._scaler_fit)
            patch(StandardScaler, "transform", self._transform)
        patch(KMeans, "fit", self._cluster)
        patch(MiniBatchKMeans, "fit", self._cluster)
        patch(BaseSampler, "fit_resample", self._resample)
        patch(CalibratedClassifierCV, "fit", self._calibrated_fit)
        patch(_SigmoidCalibration, "fit", self._sigmoid_fit)
        return self

    def __exit__(self, *exc):
        self.stack.close()

    # -- row lookup ---------------------------------------------------
    def _lookup(self, X, key):
        mapping = self.current[key]
        ids, unmatched = [], 0
        for row in np.round(np.asarray(X, dtype=float), 9) + 0.0:
            found = mapping.get(row.tobytes())
            if found is None:
                unmatched += 1
            else:
                ids.append(found)
        return ids, unmatched

    def _log(self, kind, **fields):
        self.events.append({"kind": kind, "window": self.in_window, **fields})

    # -- wrappers (each calls the real method) ---------------------------
    def _mixed_fit(self, original):
        probe = self

        def fit(inst, X, y=None):
            result = original(inst, X, y)
            probe._log("fit", ids=raw_ids(X))
            coded = inst.encode(probe.X_all)
            probe.current = {"coded": fingerprints(coded), "cluster": fingerprints(inst.expand(coded))}
            return result
        return fit

    def _imputer_fit(self, original):
        probe = self

        def fit(inst, X, *args, **kwargs):
            result = original(inst, X, *args, **kwargs)
            probe.imputer = inst
            return result
        return fit

    def _scaler_fit(self, original):
        probe = self

        def fit(inst, X, *args, **kwargs):
            result = original(inst, X, *args, **kwargs)
            probe._log("fit", ids=raw_ids(X))
            probe.muted = True
            try:
                rep = fingerprints(inst.transform(probe.imputer.transform(probe.X_all)))
            finally:
                probe.muted = False
            probe.current = {"coded": rep, "cluster": rep}
            return result
        return fit

    def _transform(self, original):
        probe = self

        def transform(inst, X, *args, **kwargs):
            if not probe.muted:
                probe._log("transform", ids=raw_ids(X))
            return original(inst, X, *args, **kwargs)
        return transform

    def _cluster(self, original):
        probe = self

        def fit(inst, X, *args, **kwargs):
            ids, unmatched = probe._lookup(X, "cluster")
            probe._log("cluster", name=type(inst).__name__, ids=ids, unmatched=unmatched)
            return original(inst, X, *args, **kwargs)
        return fit

    def _resample(self, original):
        probe = self

        def fit_resample(inst, X, y, *args, **kwargs):
            ids, unmatched = probe._lookup(X, "coded")
            probe._log("resample", name=type(inst).__name__, ids=ids, unmatched=unmatched)
            return original(inst, X, y, *args, **kwargs)
        return fit_resample

    def _calibrated_fit(self, original):
        probe = self

        def fit(inst, X, y, *args, **kwargs):
            probe._log("calibrated_fit", ids=raw_ids(X))
            probe.in_window = True
            try:
                return original(inst, X, y, *args, **kwargs)
            finally:
                probe.in_window = False
        return fit

    def _sigmoid_fit(self, original):
        probe = self

        def fit(inst, X, y, *args, **kwargs):
            probe.sigmoid_sizes.append(len(X))
            return original(inst, X, y, *args, **kwargs)
        return fit

    # -- views ----------------------------------------------------------
    def of(self, kind, window=None):
        return [e for e in self.events if e["kind"] == kind and (window is None or e["window"] == window)]


class Case:
    """Ids of the outer partitions for a table and a seed (the split the code under test also builds)."""

    def __init__(self, y, seed=42):
        split = make_split(np.asarray(y), mode="stratified", seed=seed)
        self.train, self.validation, self.test = (set(part.tolist()) for part in (split.train, split.validation, split.test))
        self.split, self.y = split, np.asarray(y)
        self.majority = {i for i in self.train if self.y[i] == 0}
        self.minority = {i for i in self.train if self.y[i] == 1}


# ERF recipes fit one resampled forest per bag; ``erf_no_smote`` keeps the default 20 bags, ``erf_one_bag`` has 1
BAGS = {"erf_no_smote": 20}

# expected clustering classes and resamplers per recipe: (cluster class names, resampler class names)
EXPECTED = {
    "csrboost": (["KMeans"], ["{smote}"]),
    "erf_one_bag": (["MiniBatchKMeans"], ["{smote}"]),
    "smote_rf": ([], ["{smote}"]),
    "erf_no_smote": (["MiniBatchKMeans"], []),
}


class Assertions:
    def assert_training_only_without_calibration(self, probe, case, model, smote):
        names_cluster, names_resample = EXPECTED[model]
        bags = BAGS.get(model, 1)
        names_cluster = names_cluster * bags
        names_resample = [name.format(smote=smote) for name in names_resample] * bags
        # preprocessing fits (the recorder and the model's own) saw exactly the training rows, each once
        fits = probe.of("fit")
        self.assertGreaterEqual(len(fits), 1)
        for event in fits:
            self.assertEqual(sorted(event["ids"]), sorted(case.train))
        clusters, resamples = probe.of("cluster"), probe.of("resample")
        self.assertEqual([e["name"] for e in clusters], names_cluster, "clustering calls")
        self.assertEqual([e["name"] for e in resamples], names_resample, "resampler calls")
        for event in clusters + resamples:
            self.assertEqual(event["unmatched"], 0, "every received row must be recognized")
            self.assertEqual(len(event["ids"]), len(set(event["ids"])), "a row was received twice")
        for event in clusters:
            # clustered retention sees every training majority row and nothing else
            # (mutation: fit on train plus validation adds validation ids and breaks this equality)
            self.assertEqual(set(event["ids"]), case.majority)
        for event in resamples:
            ids = set(event["ids"])
            self.assertTrue(ids <= case.train, f"resampler saw non-training rows: {sorted(ids - case.train)[:5]}")
            self.assertFalse(ids & (case.validation | case.test))
            if clusters:
                self.assertTrue(case.minority <= ids)  # all training minority rows are kept by clustering
            else:
                self.assertEqual(ids, case.train)  # SMOTE alone sees the whole training partition
        self.assert_the_spy_can_see_other_partitions(probe, case)

    def assert_the_spy_can_see_other_partitions(self, probe, case):
        # Non-vacuity: later in the same run the model scores validation and test rows through the same spied
        # transform, so rows outside the training partition are visible to the spy when they do appear.
        seen_after_fit = [set(e["ids"]) for e in probe.of("transform", window=False)]
        self.assertIn(case.validation, seen_after_fit)
        self.assertIn(case.test, seen_after_fit)

    def assert_calibration_uses_training_rows_only(self, probe, case, model, smote):
        names_cluster, names_resample = EXPECTED.get(model, ([], []))
        names_resample = [name.format(smote=smote) for name in names_resample]
        calibrated = probe.of("calibrated_fit")
        self.assertEqual(len(calibrated), 1)
        # the calibration wrapper itself is given exactly the training rows
        # (mutation: train plus validation rows fitted through the calibrator)
        self.assertEqual(sorted(calibrated[0]["ids"]), sorted(case.train))
        window = [e for e in probe.events if e["window"]]
        fits = [e for e in window if e["kind"] == "fit"]
        self.assertEqual(len(fits), 3, "one estimator fit per inner fold")
        held_out = []
        for event in fits:
            ids = set(event["ids"])
            self.assertTrue(ids < case.train, "an inner training fold must be a proper subset of the training rows")
            held_out.extend(sorted(case.train - ids))
        # the inner held-out folds partition the training rows: each training row is held out exactly once and
        # no validation or test row is ever used (mutation: folds built over train plus validation)
        self.assertEqual(sorted(held_out), sorted(case.train))
        for event in window:
            if event["kind"] == "transform":
                self.assertTrue(set(event["ids"]) <= case.train, "a transform saw non-training rows during calibration")
        # sigmoid calibrators are fitted on exactly those held-out predictions, one calibrator per inner fold
        self.assertEqual(len(probe.sigmoid_sizes), 3)
        self.assertEqual(probe.sigmoid_sizes, [len(case.train - set(e["ids"])) for e in fits])
        self.assertEqual(sum(probe.sigmoid_sizes), len(case.train))
        # clustering and oversampling inside each inner fold see rows of that inner training fold only
        fold_ids, clusters, resamples = None, [], []
        for event in window:
            if event["kind"] == "fit":
                fold_ids = set(event["ids"])
            elif event["kind"] in ("cluster", "resample"):
                self.assertEqual(event["unmatched"], 0)
                self.assertTrue(set(event["ids"]) <= fold_ids,
                                f"{event['name']} saw rows outside its inner training fold")
                self.assertTrue(set(event["ids"]) <= case.train)
                (clusters if event["kind"] == "cluster" else resamples).append(event["name"])
        self.assertEqual(clusters, names_cluster * 3)
        self.assertEqual(resamples, names_resample * 3)
        self.assert_the_spy_can_see_other_partitions(probe, case)


def mixed_inputs(**table):
    columns = mixed_table(with_row_id=True, **table)
    names = [name for name in columns if name != "label"]
    numeric = {"row_id", "age", "income"}
    X_all = np.empty((len(columns["label"]), len(names)), dtype=object)
    for position, name in enumerate(names):
        if name in numeric:
            X_all[:, position] = [float("nan") if value == "" else float(value) for value in columns[name]]
        else:
            X_all[:, position] = list(columns[name])
    y = [1 if value == "fraud" else 0 for value in columns["label"]]
    return columns, X_all, y


def run_mixed(columns, model, **options):
    runner, patch = patched_runner()
    with patch:
        result = train_comparison_tabular(columns, target="label", label_spec=LabelSpec("fraud", "ok"),
                                          models=(model,), **options)
    return result["outcomes"][model]


def numeric_inputs():
    columns = numeric_table(n=150, positives=45)
    X, labels = to_arrays(columns)
    X = np.column_stack([np.arange(len(X), dtype=float), X])
    y = [1 if value == "event" else 0 for value in labels]
    return X, labels, y


def run_numeric(X, labels, model, **options):
    return _fit_one(X, labels, model_name=model, positive_label="event", negative_label="clear", seed=42, **options)


class MixedWithoutCalibration(unittest.TestCase, Assertions):
    @classmethod
    def setUpClass(cls):
        cls.columns, cls.X_all, y = mixed_inputs()
        cls.case = Case(y)

    def check(self, model):
        with Probe("mixed", self.X_all) as probe:
            outcome = run_mixed(self.columns, model)
        self.assertEqual(outcome["status"], "success", outcome.get("error"))
        self.assert_training_only_without_calibration(probe, self.case, model, smote="SMOTENC")

    def test_exact_kmeans_and_smotenc_see_training_rows_only_in_csrboost(self):
        self.check("csrboost")

    def test_minibatch_kmeans_and_smotenc_see_training_rows_only_in_erf(self):
        self.check("erf_one_bag")

    def test_smotenc_alone_sees_exactly_the_training_partition(self):
        self.check("smote_rf")

    def test_clustering_without_oversampling_sees_training_majority_rows_only(self):
        self.check("erf_no_smote")


class MixedWithCalibration(unittest.TestCase, Assertions):
    @classmethod
    def setUpClass(cls):
        cls.columns, cls.X_all, y = mixed_inputs(n=150, positives=45)
        cls.case = Case(y)

    def check(self, model):
        with Probe("mixed", self.X_all) as probe:
            outcome = run_mixed(self.columns, model, calibration="sigmoid")
        self.assertEqual(outcome["status"], "success", outcome.get("error"))
        self.assert_calibration_uses_training_rows_only(probe, self.case, model, smote="SMOTENC")

    def test_sigmoid_calibration_with_exact_kmeans_and_smotenc(self):
        self.check("csrboost")

    def test_sigmoid_calibration_with_minibatch_kmeans_and_smotenc(self):
        self.check("erf_one_bag")

    def test_sigmoid_calibration_with_smotenc_alone(self):
        self.check("smote_rf")

    def test_sigmoid_calibration_of_a_model_without_resampling(self):
        self.check("logistic")


class NumericWithoutCalibration(unittest.TestCase, Assertions):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.labels, y = numeric_inputs()
        cls.case = Case(y)

    def check(self, model):
        with Probe("numeric", self.X) as probe:
            outcome = run_numeric(self.X, self.labels, model)
        self.assertEqual(outcome["status"], "success")
        self.assert_training_only_without_calibration(probe, self.case, model, smote="SMOTE")

    def test_exact_kmeans_and_smote_see_training_rows_only_in_csrboost(self):
        self.check("csrboost")

    def test_minibatch_kmeans_and_smote_see_training_rows_only_in_erf(self):
        self.check("erf_one_bag")

    def test_smote_in_the_sampling_pipeline_sees_exactly_the_training_partition(self):
        self.check("smote_rf")

    def test_clustering_without_oversampling_sees_training_majority_rows_only(self):
        self.check("erf_no_smote")


class NumericWithCalibration(unittest.TestCase, Assertions):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.labels, y = numeric_inputs()
        cls.case = Case(y)

    def check(self, model):
        with Probe("numeric", self.X) as probe:
            outcome = run_numeric(self.X, self.labels, model, calibration="sigmoid")
        self.assertEqual(outcome["status"], "success")
        self.assert_calibration_uses_training_rows_only(probe, self.case, model, smote="SMOTE")

    def test_sigmoid_calibration_with_exact_kmeans_and_smote(self):
        self.check("csrboost")

    def test_sigmoid_calibration_with_minibatch_kmeans_and_smote(self):
        self.check("erf_one_bag")

    def test_sigmoid_calibration_with_smote_in_the_sampling_pipeline(self):
        self.check("smote_rf")

    def test_sigmoid_calibration_of_a_model_without_resampling(self):
        self.check("logistic")


if __name__ == "__main__":
    unittest.main()
