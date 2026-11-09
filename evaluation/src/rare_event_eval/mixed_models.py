"""Mixed-type preprocessing, categorical-aware resampling and estimators.

These are application recipes for tables that have categorical predictors. They
are not the numeric research recipes in ``estimators.py`` and they do not
reproduce any research table. Numeric-only tables keep using the numeric recipes.

Representation, in the order it is built (everything is fitted on a training
partition only; calibration refits it inside each inner training fold):

1. ``TabularPreprocessor`` takes a two-dimensional object matrix whose columns are
   either numeric (floats, NaN for missing) or categorical (strings, ``""`` for
   missing). Numeric columns get ``SimpleImputer(strategy='median',
   keep_empty_features=True)`` and ``StandardScaler`` exactly as the numeric
   recipes do. Categorical columns get a vocabulary (the sorted distinct values of
   the training partition) and the most frequent training value for imputation
   (ties go to the smallest string).
2. ``encode`` returns the "coded" matrix: standardized numeric columns followed by
   one integer code per categorical column. A value outside the vocabulary gets the
   dedicated unseen code, equal to the vocabulary size. Nothing is refitted.
3. ``expand`` turns the coded matrix into the estimator input: the numeric columns
   plus one indicator column per vocabulary entry. The unseen code expands to an
   all-zero row for that column.

Resampling recipes (SMOTE-RF, CSRBoost-style, ERF and its ablations) work on the
coded matrix. Clustered majority retention clusters the expanded representation
(standardized numeric columns plus unweighted 0/1 indicators), so no ordering is
implied between categories. Oversampling uses imbalanced-learn ``SMOTENC`` on the
coded matrix with the code columns declared categorical: a synthetic row takes the
most frequent code among its neighbours, so it is always a member of the learned
vocabulary. No fractional code is created or rounded, and the result is checked.
Estimators then see the expanded representation of the resampled rows.

Strict failure: infeasible requests raise ``RecipeError`` with a stable ``code``;
nothing is replaced by another sampler. ``SMOTENC`` needs at least one numeric
predictor, so the resampling recipes fail with ``RESAMPLING_NO_NUMERIC_FEATURES`` on
tables whose predictors are all categorical. A categorical predictor with no observed value in the training
partition (empty vocabulary) makes SMOTENC impossible for that table and fails with
``FEATURE_EMPTY_CATEGORY_IN_TRAIN``; recipes that do not call SMOTENC are not affected.
"""
from __future__ import annotations

from numbers import Integral, Real
import math
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

EXPANSION_EXCEEDED = "EXPANSION_EXCEEDED"
RESAMPLING_INFEASIBLE = "RESAMPLING_INFEASIBLE"
RESAMPLING_NO_NUMERIC_FEATURES = "RESAMPLING_NO_NUMERIC_FEATURES"
RESAMPLING_CLUSTERING_FAILED = "RESAMPLING_CLUSTERING_FAILED"
RESAMPLING_INVALID_CATEGORY_CODE = "RESAMPLING_INVALID_CATEGORY_CODE"
FEATURE_EMPTY_CATEGORY_IN_TRAIN = "FEATURE_EMPTY_CATEGORY_IN_TRAIN"
INPUT_INVALID = "INPUT_INVALID"

UNSEEN_CATEGORY_RULE = ("A category not seen in the training partition gets the dedicated unseen code (the vocabulary size) "
                        "and expands to an all-zero indicator block for its column; no refit happens.")
CLUSTERING_REPRESENTATION = ("standardized numeric columns plus unweighted 0/1 indicator columns for each categorical "
                             "column, fitted on the training rows (application representation, not the numeric research one)")
NON_RESAMPLING_RECIPES = ("logistic", "rf_raw", "rf_weighted", "rf_weighted_1000", "balanced_rf", "hist_gradient_boosting")


class RecipeError(ValueError):
    """Typed recipe failure. ``code`` is a stable reason code; ``details`` is JSON-safe."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _matrix(X) -> np.ndarray:
    array = np.asarray(X)
    if array.dtype.kind not in "fiu":
        array = np.asarray(X, dtype=object)
    if array.ndim != 2 or array.shape[0] < 1:
        raise RecipeError(INPUT_INVALID, "X must be a two-dimensional matrix with at least one row.")
    return array


def _strings(column) -> np.ndarray:
    """Categorical cells as a string array; ``""`` marks a missing value."""
    return np.asarray(["" if value is None or (isinstance(value, float) and value != value) else str(value)
                       for value in column], dtype=str)


class TabularPreprocessor(TransformerMixin, BaseEstimator):
    """Train-fitted numeric and categorical preprocessing; see the module docstring.

    ``numeric_columns`` and ``categorical_columns`` are column positions in the
    object matrix, ``column_names`` the display names in matrix order, and
    ``max_expanded_features`` (optional) bounds the expanded width before any
    estimator is fitted.
    """

    def __init__(self, numeric_columns=(), categorical_columns=(), column_names=(), max_expanded_features=None):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.column_names = column_names
        self.max_expanded_features = max_expanded_features

    def _numeric_block(self, X):
        return np.asarray(X[:, list(self.numeric_columns)], dtype=float)

    def fit(self, X, y=None):
        X = _matrix(X)
        numeric, categorical = tuple(self.numeric_columns), tuple(self.categorical_columns)
        names = tuple(self.column_names)
        if len(numeric) + len(categorical) != X.shape[1] or sorted(numeric + categorical) != list(range(X.shape[1])):
            raise RecipeError(INPUT_INVALID, "Numeric and categorical column positions must cover every matrix column once.")
        if len(names) != X.shape[1]:
            raise RecipeError(INPUT_INVALID, "One column name per matrix column is required.")
        self.n_features_in_ = X.shape[1]
        self.n_numeric_ = len(numeric)
        self.n_categorical_ = len(categorical)
        if numeric:
            block = self._numeric_block(X)
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            self.scaler_ = StandardScaler().fit(self.imputer_.fit_transform(block))
        self.vocabularies_, self.mode_codes_, self.modes_, self.cardinalities_ = [], [], [], []
        for position in categorical:
            values = _strings(X[:, position])
            vocabulary, counts = np.unique(values[values != ""], return_counts=True)
            self.vocabularies_.append(vocabulary)
            self.cardinalities_.append(int(len(vocabulary)))
            if len(vocabulary):
                top = int(np.argmax(counts))  # first maximum: smallest string among ties
                self.mode_codes_.append(top)
                self.modes_.append(str(vocabulary[top]))
            else:
                self.mode_codes_.append(None)
                self.modes_.append(None)
        self.expanded_width_ = self.n_numeric_ + int(sum(self.cardinalities_))
        if self.max_expanded_features is not None and self.expanded_width_ > self.max_expanded_features:
            raise RecipeError(EXPANSION_EXCEEDED,
                              f"One-hot expansion would create {self.expanded_width_} columns; the limit is "
                              f"{self.max_expanded_features}. No model was fitted.",
                              expanded_width=self.expanded_width_, limit=self.max_expanded_features)
        self.feature_names_ = [str(names[position]) for position in numeric]
        for position, vocabulary in zip(categorical, self.vocabularies_):
            self.feature_names_.extend(f"{names[position]}={category}" for category in vocabulary.tolist())
        return self

    def encode(self, X) -> np.ndarray:
        """Coded matrix: standardized numeric columns, then one integer code per categorical column."""
        check_is_fitted(self, "expanded_width_")
        X = _matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise RecipeError(INPUT_INVALID, "X has a different number of columns than the fitted preprocessor.")
        parts = []
        if self.n_numeric_:
            parts.append(self.scaler_.transform(self.imputer_.transform(self._numeric_block(X))))
        for position, vocabulary, mode_code in zip(self.categorical_columns, self.vocabularies_, self.mode_codes_):
            values = _strings(X[:, position])
            size = len(vocabulary)
            codes = np.full(len(values), size, dtype=float)  # dedicated unseen code
            if size:
                index = np.minimum(np.searchsorted(vocabulary, values), size - 1)
                found = (vocabulary[index] == values) & (values != "")
                codes[found] = index[found]
            if mode_code is not None:
                codes[values == ""] = mode_code
            parts.append(codes[:, None])
        return np.hstack(parts)

    def expand(self, coded) -> np.ndarray:
        """One-hot expansion of a coded matrix; the unseen code gives an all-zero block."""
        check_is_fitted(self, "expanded_width_")
        coded = np.asarray(coded, dtype=float)
        blocks = [coded[:, :self.n_numeric_]]
        for offset, size in enumerate(self.cardinalities_):
            codes = coded[:, self.n_numeric_ + offset]
            block = np.zeros((len(coded), size))
            valid = codes < size
            block[np.flatnonzero(valid), codes[valid].astype(int)] = 1.0
            blocks.append(block)
        return np.hstack(blocks)

    def transform(self, X) -> np.ndarray:
        return self.expand(self.encode(X))

    def count_unseen(self, X) -> dict:
        """Non-missing categorical cells whose value is outside the training vocabulary, per column."""
        check_is_fitted(self, "expanded_width_")
        X = _matrix(X)
        counts = {}
        for position, vocabulary in zip(self.categorical_columns, self.vocabularies_):
            values = _strings(X[:, position])
            present = values[values != ""]
            counts[str(self.column_names[position])] = int((~np.isin(present, vocabulary)).sum()) if len(present) else 0
        return counts

    def count_missing(self, X) -> dict:
        X = _matrix(X)
        counts = {}
        for position in self.numeric_columns:
            counts[str(self.column_names[position])] = int(np.isnan(np.asarray(X[:, position], dtype=float)).sum())
        for position in self.categorical_columns:
            counts[str(self.column_names[position])] = int((_strings(X[:, position]) == "").sum())
        return counts


def resample_mixed(coded, y, pre: TabularPreprocessor, *, method, retention, seed, exact=False):
    """Categorical-aware counterpart of ``estimators._resample`` on a coded matrix.

    Returns ``(coded_resampled, y_resampled, event)``. Original (retained) rows come
    first and synthetic rows are appended, so tests can inspect them. Raises
    ``RecipeError`` for infeasible requests; there is no fallback sampler.
    """
    coded = np.asarray(coded, dtype=float)
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    event = {"method": method, "input_counts": counts.tolist(), "warnings": [], "representation": "coded matrix",
             "clustering_representation": CLUSTERING_REPRESENTATION if "clustered" in method else None}
    Xr, yr = coded, y
    if method in ("clustered_smote", "clustered"):
        minor_rows, major_rows = np.flatnonzero(y == minority), np.flatnonzero(y == majority)
        k = min(len(minor_rows), len(major_rows))
        if k < 1:
            raise RecipeError(RESAMPLING_INFEASIBLE, "Clustered retention needs at least one row in each class; no fallback was used.")
        representation = pre.expand(coded[major_rows])
        clusterer = (KMeans(n_clusters=k, n_init=10, random_state=seed) if exact else
                     MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=100, batch_size=1024, random_state=seed))
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                clusters = clusterer.fit_predict(representation)
        except ValueError as exc:
            raise RecipeError(RESAMPLING_CLUSTERING_FAILED, f"Clustering failed: {exc}. No fallback was used.") from exc
        event["warnings"] = [str(item.message) for item in captured]
        rng = np.random.RandomState(seed)
        kept = []
        for label in np.unique(clusters):
            members = major_rows[clusters == label]
            count = max(1, math.ceil(len(members) * retention))
            kept.extend(rng.choice(members, count, replace=False).tolist())
        indices = np.concatenate([minor_rows, np.asarray(kept, dtype=int)])
        Xr, yr = coded[indices], y[indices]
        event.update(requested_clusters=k, actual_clusters=int(np.unique(clusters).size),
                     retained_counts=np.bincount(yr, minlength=2).tolist())
        if event["actual_clusters"] < k:
            event["warnings"].append(f"{event['actual_clusters']} non-empty clusters were found for {k} requested; "
                                     "each non-empty cluster still retains at least one row.")
    if method in ("clustered_smote", "smote"):
        retained = np.bincount(yr, minlength=2)
        if retained[0] != retained[1]:
            if retained.min() < 2:
                raise RecipeError(RESAMPLING_INFEASIBLE,
                                  "SMOTE requires at least two rows in the class to be augmented; no fallback was used.",
                                  retained_counts=retained.tolist())
            try:
                from imblearn.over_sampling import SMOTE, SMOTENC
            except ImportError as exc:
                raise ValueError("This model requires optional training dependencies. Install the package with [training].") from exc
            neighbors = min(5, int(retained.min()) - 1)
            before = len(yr)
            if pre.n_categorical_:
                if pre.n_numeric_ == 0:
                    raise RecipeError(RESAMPLING_NO_NUMERIC_FEATURES,
                                      "SMOTENC needs at least one numeric predictor; all predictors are categorical. "
                                      "No other sampler was substituted.")
                empty = [str(pre.column_names[position]) for position, size in zip(pre.categorical_columns, pre.cardinalities_)
                         if size == 0]
                if empty:
                    shown = ", ".join(repr(name) for name in empty[:5]) + (f" and {len(empty) - 5} more" if len(empty) > 5 else "")
                    raise RecipeError(FEATURE_EMPTY_CATEGORY_IN_TRAIN,
                                      f"Categorical predictor {shown} has no observed value in the training partition, so it has "
                                      "no category vocabulary and SMOTENC cannot oversample. Exclude the column or choose a "
                                      "recipe without SMOTE. No other sampler was substituted.",
                                      columns=empty[:50], column_count=len(empty))
                categorical = list(range(pre.n_numeric_, pre.n_numeric_ + pre.n_categorical_))
                sampler, variant = SMOTENC(categorical_features=categorical, k_neighbors=neighbors, random_state=seed), "SMOTENC"
            else:
                sampler, variant = SMOTE(k_neighbors=neighbors, random_state=seed), "SMOTE"
            Xr, yr = sampler.fit_resample(Xr, yr)
            Xr = np.asarray(Xr, dtype=float)
            if pre.n_categorical_:
                new = Xr[before:, pre.n_numeric_:]
                limits = np.asarray(pre.cardinalities_, dtype=float)
                if len(new) and not (np.isfinite(new).all() and (new == np.round(new)).all() and ((new >= 0) & (new < limits)).all()):
                    raise RecipeError(RESAMPLING_INVALID_CATEGORY_CODE,
                                      "A synthetic row has a category code outside the learned vocabulary. The result was rejected.")
            event.update(smote_neighbors=neighbors, smote_variant=variant, synthetic_rows=int(len(yr) - before),
                         synthetic_category_codes_in_vocabulary=True)
        else:
            event.update(smote_neighbors=None, smote_variant=None, synthetic_rows=0)
            event["warnings"].append("Retained classes already balanced; no synthetic rows needed.")
    event["output_counts"] = np.bincount(yr, minlength=2).tolist()
    return Xr, yr, event


def _prepare_fit(estimator, X, y):
    X = _matrix(X)
    classes, binary = np.unique(np.asarray(y), return_inverse=True)
    if len(classes) != 2:
        raise ValueError(f"Only binary classification is supported. Training requires 2 classes; got {len(classes)} class(es).")
    pre = TabularPreprocessor(numeric_columns=estimator.numeric_columns, categorical_columns=estimator.categorical_columns,
                              column_names=estimator.column_names,
                              max_expanded_features=estimator.max_expanded_features).fit(X)
    return X, pre, pre.encode(X), classes, binary


class _MixedClassifier(ClassifierMixin, BaseEstimator):
    """Shared plumbing: raw object matrix in, expanded representation to the final model."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _models(self):
        raise NotImplementedError

    def predict_proba(self, X):
        check_is_fitted(self, "preprocessor_")
        Z = self.preprocessor_.transform(_matrix(X))
        return np.mean([model.predict_proba(Z) for model in self._models()], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class MixedSmoteRFClassifier(_MixedClassifier):
    """SMOTENC on the coded matrix, then an unweighted random forest on the expanded representation."""

    def __init__(self, numeric_columns=(), categorical_columns=(), column_names=(), max_expanded_features=None,
                 n_estimators=100, random_state=42, n_jobs=1):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.column_names = column_names
        self.max_expanded_features = max_expanded_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        _integer(self.n_estimators, "n_estimators")
        X, pre, coded, classes, binary = _prepare_fit(self, X, y)
        seed = int(self.random_state)
        Xr, yr, event = resample_mixed(coded, binary, pre, method="smote", retention=1.0, seed=seed)
        model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=seed, n_jobs=self.n_jobs)
        model.fit(pre.expand(Xr), yr)
        self.n_features_in_ = X.shape[1]
        self.classes_, self.preprocessor_, self.model_, self.resampling_events_ = classes, pre, model, [event]
        return self

    def _models(self):
        return [self.model_]


class MixedCSRBoostClassifier(_MixedClassifier):
    """Exact-KMeans clustered retention, SMOTENC, then AdaBoost over unlimited-depth trees (mixed application recipe)."""

    def __init__(self, numeric_columns=(), categorical_columns=(), column_names=(), max_expanded_features=None,
                 n_estimators=50, retention=.5, random_state=42):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.column_names = column_names
        self.max_expanded_features = max_expanded_features
        self.n_estimators = n_estimators
        self.retention = retention
        self.random_state = random_state

    def fit(self, X, y):
        _integer(self.n_estimators, "n_estimators")
        if not isinstance(self.retention, Real) or isinstance(self.retention, bool) or not 0 < self.retention <= 1:
            raise ValueError("retention must lie in (0, 1].")
        X, pre, coded, classes, binary = _prepare_fit(self, X, y)
        seed = int(self.random_state)
        Xr, yr, event = resample_mixed(coded, binary, pre, method="clustered_smote", retention=self.retention,
                                       seed=seed, exact=True)
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=seed),
                                   n_estimators=self.n_estimators, random_state=seed)
        model.fit(pre.expand(Xr), yr)
        self.n_features_in_ = X.shape[1]
        self.classes_, self.preprocessor_, self.model_, self.resampling_events_ = classes, pre, model, [event]
        return self

    def _models(self):
        return [self.model_]


class MixedERFClassifier(_MixedClassifier):
    """Bagged resampled forests on mixed inputs: per bag clustered retention and SMOTENC (mixed application recipe).

    Same capacity and seed rule as ``ERFClassifier`` (bag seed = seed + bag * 7919).
    The preprocessor is fitted once on the training rows given to ``fit``.
    """

    def __init__(self, numeric_columns=(), categorical_columns=(), column_names=(), max_expanded_features=None,
                 n_bags=20, trees_per_bag=50, retention=.5, resampling="clustered_smote", random_state=42, n_jobs=1):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.column_names = column_names
        self.max_expanded_features = max_expanded_features
        self.n_bags = n_bags
        self.trees_per_bag = trees_per_bag
        self.retention = retention
        self.resampling = resampling
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        _integer(self.n_bags, "n_bags")
        _integer(self.trees_per_bag, "trees_per_bag")
        if not isinstance(self.retention, Real) or isinstance(self.retention, bool) or not 0 < self.retention <= 1:
            raise ValueError("retention must lie in (0, 1].")
        if self.resampling not in ("clustered_smote", "smote", "clustered", "none"):
            raise ValueError("Unsupported resampling method.")
        X, pre, coded, classes, binary = _prepare_fit(self, X, y)
        base_seed = int(self.random_state)
        models, events = [], []
        for bag in range(self.n_bags):
            seed = (base_seed + bag * 7919) % (2**32 - 1)
            Xb, yb, event = resample_mixed(coded, binary, pre, method=self.resampling, retention=self.retention, seed=seed)
            model = RandomForestClassifier(n_estimators=self.trees_per_bag, random_state=seed, n_jobs=self.n_jobs)
            model.fit(pre.expand(Xb), yb)
            models.append(model)
            events.append(event)
        self.n_features_in_ = X.shape[1]
        self.classes_, self.preprocessor_, self.models_, self.resampling_events_ = classes, pre, models, events
        return self

    def _models(self):
        return self.models_


def make_mixed_model(name, *, seed=42, minority_count=10, numeric_columns, categorical_columns, column_names,
                     max_expanded_features=None):
    """Build the mixed-type pipeline for a package model id.

    The final estimators are taken from ``training.make_model``, so capacity,
    class weights and seeds cannot drift from the numeric recipes. The input is the
    object matrix described in the module docstring.
    """
    from .training import make_model
    if name == "smote_rf" and minority_count < 2:
        raise RecipeError(RESAMPLING_INFEASIBLE, "SMOTE requires at least two training rows in the smaller class; no fallback was used.",
                          minority_count=int(minority_count))
    base = make_model(name, seed=seed, minority_count=max(int(minority_count), 2))
    reference = base.named_steps["model"]
    columns = dict(numeric_columns=tuple(numeric_columns), categorical_columns=tuple(categorical_columns),
                   column_names=tuple(column_names), max_expanded_features=max_expanded_features)
    if name in NON_RESAMPLING_RECIPES:
        return Pipeline([("pre", TabularPreprocessor(**columns)), ("model", reference)])
    if name == "smote_rf":
        estimator = MixedSmoteRFClassifier(**columns, n_estimators=reference.n_estimators, random_state=seed, n_jobs=1)
    elif name == "csrboost":
        estimator = MixedCSRBoostClassifier(**columns, n_estimators=reference.n_estimators,
                                            retention=reference.retention, random_state=seed)
    elif name in ("erf", "erf_no_clustering", "erf_no_smote", "erf_one_bag"):
        estimator = MixedERFClassifier(**columns, n_bags=reference.n_bags, trees_per_bag=reference.trees_per_bag,
                                       retention=reference.retention, resampling=reference.resampling,
                                       random_state=seed, n_jobs=1)
    else:
        raise ValueError("Unknown model name.")
    return Pipeline([("model", estimator)])


def preprocessor_of(model):
    """The fitted ``TabularPreprocessor`` of a fitted mixed pipeline."""
    steps = model.named_steps
    return steps["pre"] if "pre" in steps else steps["model"].preprocessor_
