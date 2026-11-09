"""Cloneable numeric CSRBoost-style and ERF estimators.

These are new controlled implementations. They do not reproduce the historical
runner byte-for-byte. Learned imputation/scaling belong in a training pipeline.
Resampling errors are raised. There is no silent unresampled fallback.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:
    raise ImportError("Training requires the optional dependencies: install this package with [training].") from exc


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _resample(X, y, *, method, retention, seed, exact=False):
    counts = np.bincount(y, minlength=2)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    event = {"method": method, "input_counts": counts.tolist(), "warnings": []}
    Xr, yr = X, y
    if method in ("clustered_smote", "clustered"):
        minor_rows, major_rows = np.flatnonzero(y == minority), np.flatnonzero(y == majority)
        k = min(len(minor_rows), len(major_rows))
        clusterer = (KMeans(n_clusters=k, n_init=10, random_state=seed) if exact else
                     MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=100,
                                     batch_size=1024, random_state=seed))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            clusters = clusterer.fit_predict(X[major_rows])
        event["warnings"] = [str(item.message) for item in captured]
        rng = np.random.RandomState(seed)
        kept = []
        for label in np.unique(clusters):
            members = major_rows[clusters == label]
            count = max(1, math.ceil(len(members) * retention))
            kept.extend(rng.choice(members, count, replace=False).tolist())
        indices = np.concatenate([minor_rows, np.asarray(kept, dtype=int)])
        Xr, yr = X[indices], y[indices]
        event.update(requested_clusters=k, actual_clusters=int(np.unique(clusters).size),
                     retained_counts=np.bincount(yr, minlength=2).tolist())
    if method in ("clustered_smote", "smote"):
        retained = np.bincount(yr, minlength=2)
        if retained[0] != retained[1]:
            if retained.min() < 2:
                raise ValueError("SMOTE requires at least two rows in the class to be augmented; no fallback was used.")
            neighbors = min(5, int(retained.min()) - 1)
            Xr, yr = SMOTE(k_neighbors=neighbors, random_state=seed).fit_resample(Xr, yr)
            event["smote_neighbors"] = neighbors
        else:
            event["smote_neighbors"] = None
            event["warnings"].append("Retained classes already balanced; no synthetic rows needed.")
    event["output_counts"] = np.bincount(yr, minlength=2).tolist()
    return Xr, yr, event


class ERFClassifier(ClassifierMixin, BaseEstimator):
    """Bagged resampled forests, numeric binary classification only.

    Default: 20 bags x 50 trees, 50% per-cluster retention and SMOTE.
    resampling supports controlled ablations. n_jobs defaults to one.
    """

    def __init__(self, n_bags=20, trees_per_bag=50, retention=.5,
                 resampling="clustered_smote", random_state=42, n_jobs=1):
        self.n_bags = n_bags
        self.trees_per_bag = trees_per_bag
        self.retention = retention
        self.resampling = resampling
        self.random_state = random_state
        self.n_jobs = n_jobs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        _integer(self.n_bags, "n_bags")
        _integer(self.trees_per_bag, "trees_per_bag")
        if not isinstance(self.retention, Real) or isinstance(self.retention, bool) or not 0 < self.retention <= 1:
            raise ValueError("retention must lie in (0, 1].")
        if self.resampling not in ("clustered_smote", "smote", "clustered", "none"):
            raise ValueError("Unsupported resampling method.")
        if self.n_jobs is not None and (isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, Integral) or self.n_jobs == 0):
            raise ValueError("n_jobs must be a nonzero integer or None.")
        X, y = validate_data(self, X, y, dtype=np.float64, accept_sparse=False)
        check_classification_targets(y)
        classes, binary = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError(f"Only binary classification is supported. Training requires 2 classes; got {len(classes)} class(es).")
        rng = check_random_state(self.random_state)
        base_seed = (int(self.random_state) if isinstance(self.random_state, Integral)
                     else int(rng.randint(0, 2**30)))
        models, events = [], []
        for bag in range(self.n_bags):
            seed = (base_seed + bag * 7919) % (2**32 - 1)
            Xb, yb, event = _resample(X, binary, method=self.resampling,
                                      retention=self.retention, seed=seed)
            model = RandomForestClassifier(n_estimators=self.trees_per_bag,
                                           random_state=seed, n_jobs=self.n_jobs)
            model.fit(Xb, yb)
            models.append(model)
            events.append(event)
        self.classes_, self.models_, self.resampling_events_ = classes, models, events
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "models_")
        X = validate_data(self, X, reset=False, dtype=np.float64, accept_sparse=False)
        return np.mean([model.predict_proba(X) for model in self.models_], axis=0)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class CSRBoostClassifier(ClassifierMixin, BaseEstimator):
    """New numeric CSRBoost-style KMeans/SMOTE/AdaBoost pipeline component."""

    def __init__(self, n_estimators=50, retention=.5, random_state=42):
        self.n_estimators = n_estimators
        self.retention = retention
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        _integer(self.n_estimators, "n_estimators")
        if not isinstance(self.retention, Real) or isinstance(self.retention, bool) or not 0 < self.retention <= 1:
            raise ValueError("retention must lie in (0, 1].")
        X, y = validate_data(self, X, y, dtype=np.float64, accept_sparse=False)
        check_classification_targets(y)
        classes, binary = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError(f"Only binary classification is supported. Training requires 2 classes; got {len(classes)} class(es).")
        seed = int(check_random_state(self.random_state).randint(0, 2**30))
        if isinstance(self.random_state, Integral):
            seed = int(self.random_state)
        Xb, yb, event = _resample(X, binary, method="clustered_smote", retention=self.retention,
                                  seed=seed, exact=True)
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=seed),
                                    n_estimators=self.n_estimators, random_state=seed)
        model.fit(Xb, yb)
        self.classes_, self.model_, self.resampling_events_ = classes, model, [event]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = validate_data(self, X, reset=False, dtype=np.float64, accept_sparse=False)
        return self.model_.predict_proba(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
