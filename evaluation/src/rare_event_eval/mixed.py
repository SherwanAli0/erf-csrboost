"""Mixed-type training adapter: numeric, categorical and missing predictors.

``train_comparison_tabular`` is the application entry point for tables that may have
categorical predictors. It runs, in this order:

1. ``tabular.profile_table`` (byte, row, column, cardinality and one-hot width limits
   are checked first);
2. ``labels.resolve_labels`` (explicit positive label, optional one-versus-rest,
   explicit missing-target policy) and a record of every excluded column and row;
3. ``splits.make_split`` once, in the parent, with exact achieved counts. An
   infeasible request raises ``MixedError`` (``SPLIT_INFEASIBLE``); it is never
   reported as the requested split and no other split is substituted;
4. one disposable worker subprocess per model (same isolation, thread limits and
   timeout as the numeric path). The worker receives typed columns, numeric float
   arrays and ``'U'`` string arrays in an ``.npz`` written and read with
   ``allow_pickle=False``, plus a ``config.json`` of column kinds, and fits all
   preprocessing, resampling and calibration on the training partition only;
5. a validation-selected threshold (``policies.select_threshold``) and test-partition
   evaluation (``core.evaluate_predictions``);
6. a ``training-1.1`` report using the fields documented in ``docs/report-contract.md``.

Recipe choice. A table whose predictors are all numeric runs the unchanged numeric
recipes (``training.make_model``), so its metrics equal those of
``training.train_comparison`` for the same seed. A table with at least one categorical
predictor runs the mixed variants in ``mixed_models`` (own ``recipe_id`` values in
``registry.MIXED_REGISTRY``). Only the ``strict`` fallback policy exists: an
infeasible recipe is a failed model with a reason code, never another recipe labelled
as success.

pandas is not imported by this module. DataFrames are read by duck typing.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
import time
import warnings
import zipfile

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from threadpoolctl import threadpool_limits

from .core import CostAssumptions, evaluate_predictions
from .labels import LabelSpec, resolve_labels
from .mixed_models import (CLUSTERING_REPRESENTATION, UNSEEN_CATEGORY_RULE, RecipeError, TabularPreprocessor,
                           make_mixed_model, preprocessor_of)
from .policies import select_threshold
from .registry import REGISTRY, get_mixed_spec
from .splits import feature_groups, make_split
from .tabular import (ParsedTable, TabularError, TabularLimits, _as_columns, _is_missing_cell, _number,
                      parse_table_csv, profile_table)
from .training import (DEFAULT_MODELS, MODEL_NAMES, PREPROCESSING_DESCRIPTION, TRAINING_SCHEMA, _inner_splits,
                       _peak_rss, failure_reason_code, make_model, run_worker_directory)

ADAPTER_VERSION = "1"
SPLIT_MODES = ("stratified", "group", "feature_group", "chronological")
CALIBRATIONS = ("none", "sigmoid")
MIXED_PREPROCESSING_DESCRIPTION = (
    "numeric: median imputation and standardization; categorical: training vocabulary, most frequent training value for "
    "missing cells, one-hot expansion with unseen categories as all-zero indicators; all fitted on the training partition only")
DEFAULT_DESCRIPTION = "User-supplied tabular training data; provenance is not independently verified."
DESCRIPTION_SUFFIX = " Fresh fit with separate evaluation partitions."
MAX_DESCRIPTION_CHARACTERS = 450

OPTION_INVALID = "OPTION_INVALID"
PREDICTOR_UNKNOWN = "PREDICTOR_UNKNOWN"
NON_FINITE_VALUE = "NON_FINITE_VALUE"
GROUP_MISSING = "GROUP_MISSING"
TIME_INVALID = "TIME_INVALID"
SPLIT_INFEASIBLE = "SPLIT_INFEASIBLE"
SPLIT_UNSUPPORTED = "SPLIT_UNSUPPORTED"
CALIBRATION_INCOMPATIBLE = "CALIBRATION_INCOMPATIBLE"
CALIBRATION_INFEASIBLE = "CALIBRATION_INFEASIBLE"
PANDAS_REQUIRED = "PANDAS_REQUIRED"
CATEGORICAL_REJECTED = "CATEGORICAL_REJECTED"
ERROR_CODES = (CATEGORICAL_REJECTED, OPTION_INVALID, PREDICTOR_UNKNOWN, NON_FINITE_VALUE, GROUP_MISSING, TIME_INVALID, SPLIT_INFEASIBLE,
               SPLIT_UNSUPPORTED, CALIBRATION_INCOMPATIBLE, CALIBRATION_INFEASIBLE, PANDAS_REQUIRED)


class MixedError(TabularError):
    """Typed adapter failure for a whole request. ``code`` is stable; ``details`` is JSON-safe."""


# ---------------------------------------------------------------- column typing (parent side)

def _column_kind(record: dict) -> str:
    kind = record["kind"]
    if kind == "categorical":
        return "categorical"
    if kind == "constant":
        return "categorical" if record["value_type"] == "categorical" else "numeric"
    return "numeric"  # numeric and empty (all missing) columns


def _numeric_array(name, values, rows) -> np.ndarray:
    out = np.empty(len(rows), dtype=float)
    for position, row in enumerate(rows):
        value = values[row]
        if _is_missing_cell(value):
            out[position] = np.nan
            continue
        number = _number(value)
        if number is None or math.isinf(number):
            raise MixedError(NON_FINITE_VALUE, f"Column {name!r} has a value that is not a finite number (row {row}). "
                             "Infinity is rejected; nothing was changed.", column=name, row_index=int(row))
        out[position] = number  # NaN stays NaN and is handled as missing
    return out


def _categorical_array(values, rows) -> np.ndarray:
    return np.asarray(["" if _is_missing_cell(values[row]) else (values[row] if isinstance(values[row], str) else str(values[row]))
                       for row in rows], dtype=str)


def _group_array(name, values, rows) -> np.ndarray:
    out = []
    for row in rows:
        if _is_missing_cell(values[row]):
            raise MixedError(GROUP_MISSING, f"Group column {name!r} has a missing value (row {row}). Group identifiers "
                             "cannot be empty.", column=name, row_index=int(row))
        out.append(values[row] if isinstance(values[row], str) else str(values[row]))
    return np.asarray(out, dtype=str)


def _parse_time(value):
    number = _number(value)
    if number is not None:
        return number
    if isinstance(value, datetime):
        return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value).timestamp()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed).timestamp()
    return None


def _time_array(name, values, rows) -> np.ndarray:
    out = np.empty(len(rows), dtype=float)
    for position, row in enumerate(rows):
        parsed = None if _is_missing_cell(values[row]) else _parse_time(values[row])
        if parsed is None or not math.isfinite(parsed):
            raise MixedError(TIME_INVALID, f"Time column {name!r} has a missing or unparseable value (row {row}). Use numbers "
                             "or ISO dates/timestamps; naive dates are read as UTC.", column=name, row_index=int(row))
        out[position] = parsed
    return out


def _mixed_feature_groups(arrays, specs) -> np.ndarray:
    """Exact predictor-pattern groups. Numeric-only tables use ``splits.feature_groups`` unchanged."""
    if all(spec["kind"] == "numeric" for spec in specs):
        return feature_groups(np.column_stack([arrays[spec["key"]] for spec in specs]))
    keys = []
    for row in range(len(next(iter(arrays.values())))):
        cells = []
        for spec in specs:
            value = arrays[spec["key"]][row]
            if spec["kind"] == "numeric":
                cells.append(None if np.isnan(value) else (0.0 if value == 0 else float(value)))
            else:
                cells.append(str(value))
        keys.append(sha256(json.dumps(cells, separators=(",", ":")).encode()).hexdigest())
    return np.asarray(keys)


def _split_diagnostics(patterns, y, split, *, groups=None, times=None) -> dict:
    """Same record as ``splits.split_diagnostics`` but from precomputed predictor-pattern keys."""
    y = np.asarray(y)
    indices = {name: getattr(split, name) for name in ("train", "validation", "test")}
    train_patterns = set(patterns[split.train].tolist())
    label_sets = {}
    for pattern, label in zip(patterns, y):
        label_sets.setdefault(pattern, set()).add(str(label))
    unique, counts = np.unique(patterns, return_counts=True)
    result = {
        "mode": split.mode, "row_indices_disjoint": True,
        "partitions": {name: {"rows": len(part), "positive_rows": int(y[part].sum()),
                              "indices_sha256": sha256(np.sort(part).astype("<i8").tobytes()).hexdigest()}
                       for name, part in indices.items()},
        "duplicate_feature_groups": int((counts > 1).sum()),
        "rows_in_duplicate_groups": int(counts[counts > 1].sum()),
        "conflicting_feature_patterns": sum(len(labels) > 1 for labels in label_sets.values()),
        "test_rows_matching_training_features": sum(pattern in train_patterns for pattern in patterns[split.test]),
        "group_overlap_checked": groups is not None and split.mode in ("group", "feature_group"),
        "interpretation": "Feature-pattern overlap is a diagnostic, not automatic proof of leakage. Grouping changes the generalization question.",
    }
    if times is not None:
        result["time_ranges"] = {name: [str(np.min(np.asarray(times)[part])), str(np.max(np.asarray(times)[part]))]
                                 for name, part in indices.items()}
    return result


def _write_arrays(path, arrays) -> None:
    """Write an ``.npz`` of numeric and 'U' arrays; object arrays are refused (nothing is pickled)."""
    arrays = {name: np.asarray(array) for name, array in arrays.items()}
    for name, array in arrays.items():
        if array.dtype.kind not in "biufU":
            raise MixedError(OPTION_INVALID, f"Worker input {name!r} has dtype {array.dtype}; only numeric and string arrays are allowed.")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, array in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)


# ---------------------------------------------------------------- worker side

def _plain(value):
    if isinstance(value, (str, int, float, bool, type(None))) and not (isinstance(value, float) and not math.isfinite(value)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, (str, int)) and not isinstance(item, bool) for item in value):
        return list(value)
    return str(value)


def _constructed_steps(model) -> list:
    inner = model.estimator if isinstance(model, CalibratedClassifierCV) else model
    steps = [{"step": name, "class": type(step).__name__,
              "params": {key: _plain(value) for key, value in sorted(step.get_params(deep=False).items())}}
             for name, step in inner.steps]
    if isinstance(model, CalibratedClassifierCV):
        steps.append({"step": "calibration", "class": "CalibratedClassifierCV",
                      "params": {"method": "sigmoid", "ensemble": True, "cv": "3 training-only inner folds", "n_jobs": 1}})
    return steps


def _preprocessing_record(recorder: TabularPreprocessor, partitions: dict, specs: list, limits: dict) -> dict:
    """Alignment and unseen/missing-value record from a preprocessor fitted on the training rows only."""
    shapes, unseen, missing = {}, {}, {}
    for name, rows in partitions.items():
        matrix = recorder.transform(rows)
        shapes[name] = [int(matrix.shape[0]), int(matrix.shape[1])]
        unseen[name] = recorder.count_unseen(rows)
        missing[name] = recorder.count_missing(rows)
    names = list(recorder.feature_names_)
    if any(shape[1] != len(names) for shape in shapes.values()):
        raise RecipeError("INPUT_INVALID", "Transformed partitions do not have identical columns.")
    categorical = {}
    for position, vocabulary, mode, size in zip(recorder.categorical_columns, recorder.vocabularies_, recorder.modes_,
                                                recorder.cardinalities_):
        column = str(recorder.column_names[position])
        categorical[column] = {"vocabulary_size": size, "most_frequent_training_value": mode,
                               "indicator_columns": [f"{column}={value}" for value in vocabulary.tolist()],
                               "unseen_category_cells": {name: unseen[name][column] for name in partitions}}
    return {
        "fitted_on": "training partition only", "train_rows": shapes["train"][0],
        "predictors": [{"name": spec["name"], "kind": spec["kind"],
                        "imputation": "median" if spec["kind"] == "numeric" else "most_frequent_training_value",
                        "missing_cells": {name: missing[name][spec["name"]] for name in partitions}} for spec in specs],
        "numeric_columns": [spec["name"] for spec in specs if spec["kind"] == "numeric"],
        "categorical_columns": categorical,
        "unseen_category_rule": UNSEEN_CATEGORY_RULE,
        "expanded_feature_names": names, "expanded_feature_count": len(names),
        "alignment": {"partition_shapes": shapes, "identical_columns_across_partitions": True,
                      "columns_sha256": sha256(json.dumps(names).encode()).hexdigest(),
                      "column_order": "numeric predictors in table order, then one indicator block per categorical predictor in table order"},
        "limits": limits,
    }


def fit_tabular_model(data, config) -> dict:
    """Worker payload for one model on typed columns; see the module docstring.

    ``data`` maps array names to numpy arrays (``labels``, ``train``, ``validation``, ``test``, optional
    ``groups`` and ``times``, and one array per predictor named by ``config['columns'][i]['key']``).
    Raw scores stay local for paired comparisons.
    """
    specs = config["columns"]
    model_name, seed = config["model_name"], config["seed"]
    positive, negative = config["positive_label"], config["negative_label"]
    split_mode, calibration = config["split_mode"], config["calibration"]
    limits = config["limits"]
    costs = config.get("costs")
    if isinstance(costs, dict):  # the worker entry point has already converted it; direct callers may pass a mapping
        costs = CostAssumptions(**costs)
    if model_name not in MODEL_NAMES or calibration not in CALIBRATIONS or split_mode not in SPLIT_MODES:
        raise ValueError("Unknown model, split or calibration method.")
    labels = np.asarray(data["labels"])
    if labels.ndim != 1 or labels.dtype.kind != "U" or not specs or any(spec["kind"] not in ("numeric", "categorical") for spec in specs):
        raise ValueError("Training labels must be exact strings, with no missing values.")
    rows = len(labels)
    if not 10 <= rows <= limits["max_rows"] or any(len(data[spec["key"]]) != rows for spec in specs):
        raise ValueError("Training requires at least 10 aligned rows within the configured row limit; no sampling is performed.")
    if positive == negative or not np.isin(labels, [positive, negative]).all():
        raise ValueError("Invalid numeric features or explicit binary labels.")
    fractions = list(config["review_fractions"])
    if any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in fractions):
        raise ValueError("Review fractions must be finite values from 0 to 1.")
    y = (labels == positive).astype(int)
    partitions = {name: np.asarray(data[name], dtype=np.int64) for name in ("train", "validation", "test")}
    joined = np.concatenate(list(partitions.values()))
    if len(joined) != rows or len(np.unique(joined)) != rows or joined.min() < 0 or joined.max() >= rows:
        raise ValueError("Split indices are not a disjoint partition of the input.")
    if any(len(np.unique(y[part])) != 2 for part in partitions.values()):
        raise ValueError("Every partition must contain both classes. Requested split is infeasible; no fallback was used.")
    numeric_only = all(spec["kind"] == "numeric" for spec in specs)
    if numeric_only:
        X = np.column_stack([np.asarray(data[spec["key"]], dtype=float) for spec in specs])
        if np.isinf(X).any():
            raise ValueError("Invalid numeric features or explicit binary labels.")
    else:
        X = np.empty((rows, len(specs)), dtype=object)
        for position, spec in enumerate(specs):
            X[:, position] = np.asarray(data[spec["key"]]).astype(object)
    names = tuple(spec["name"] for spec in specs)
    numeric_columns = tuple(i for i, spec in enumerate(specs) if spec["kind"] == "numeric")
    categorical_columns = tuple(i for i, spec in enumerate(specs) if spec["kind"] == "categorical")
    train, validation, test = partitions["train"], partitions["validation"], partitions["test"]
    Xtr, ytr = X[train], y[train]

    # Record the train-only preprocessing (and enforce the expansion limit) before any model is built or fitted.
    recorder = TabularPreprocessor(numeric_columns=numeric_columns, categorical_columns=categorical_columns,
                                   column_names=names, max_expanded_features=limits["max_expanded_features"]).fit(Xtr)
    record = _preprocessing_record(recorder, {"train": Xtr, "validation": X[validation], "test": X[test]},
                                   specs, limits)
    minority = int(np.bincount(ytr).min())
    if numeric_only:
        model = make_model(model_name, seed=seed, minority_count=minority)
    else:
        model = make_mixed_model(model_name, seed=seed, minority_count=minority, numeric_columns=numeric_columns,
                                 categorical_columns=categorical_columns, column_names=names,
                                 max_expanded_features=limits["max_expanded_features"])
    if calibration == "sigmoid":
        if split_mode == "chronological":
            raise RecipeError(CALIBRATION_INCOMPATIBLE, "Sigmoid calibration with a chronological split is refused: ordered inner "
                              "folds can split equal timestamps. No other combination was substituted.")
        train_groups = np.asarray(data["groups"])[train] if "groups" in data else None
        cv = _inner_splits(Xtr, ytr, split_mode, train_groups, seed)
        model = CalibratedClassifierCV(model, method="sigmoid", cv=cv, n_jobs=1, ensemble=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - started
        validation_scores = model.predict_proba(X[validation])[:, list(model.classes_).index(1)]
        selection = select_threshold(y[validation], validation_scores, costs=costs)
        test_scores = model.predict_proba(X[test])[:, list(model.classes_).index(1)]
    budgets = sorted(set(math.ceil(float(fraction) * len(test)) for fraction in fractions))
    description = config.get("description") or "Fresh model fit; held-out predictions from the declared versioned protocol."
    report = evaluate_predictions(labels[test], {model_name: test_scores}, positive_label=positive, negative_label=negative,
                                  score_kind="probability", threshold=selection["threshold"], review_budgets=budgets,
                                  costs=costs, evaluation_role="test", data_description=description)
    report["models"][model_name]["threshold_selection"] = selection
    report["execution_mode"] = "fresh_training"
    report["policies"]["selection"] = "Per-model threshold selected on validation data only; test labels excluded from selection."
    report["checks"].update(training_preprocessing="fitted_on_training_partitions",
                            train_test_overlap="row_indices_disjoint", temporal_or_group_independence="checked_for_requested_split")
    if not numeric_only:
        report["checks"]["categorical_encoding"] = "training vocabulary; unseen categories encoded as all-zero indicators"
    report["checks"]["training_leakage"] = "not_certified: data availability and domain semantics still require review"
    events = []
    if calibration == "none":
        estimator = model.named_steps["model"]
        events = getattr(estimator, "resampling_events_", [])
        if not numeric_only and list(preprocessor_of(model).feature_names_) != list(recorder.feature_names_):
            raise RecipeError("INPUT_INVALID", "The model's preprocessing differs from the recorded training preprocessing.")
    metadata = {
        "model": model_name, "seed": seed, "split_mode": split_mode, "calibration": calibration,
        "preprocessing": PREPROCESSING_DESCRIPTION if numeric_only else MIXED_PREPROCESSING_DESCRIPTION,
        "recipe_variant": "numeric" if numeric_only else "mixed",
        "preprocessing_record": record,
        "fit_seconds": fit_seconds, "fit_and_evaluation_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": _peak_rss(), "memory_scope": "worker process peak including interpreter and libraries",
        "threads": 1, "warnings": [str(item.message) for item in captured], "resampling_events": events,
        "threshold_selection": selection, "constructed_steps": _constructed_steps(model),
    }
    return {"status": "success", "report": report, "execution": metadata,
            "test_scores": test_scores.tolist(), "test_truth": y[test].tolist(), "test_indices": test.tolist(),
            "test_groups": None}


# ---------------------------------------------------------------- parent side

def _import_pandas():
    import pandas
    return pandas


def _require_pandas():
    """Availability gate for the optional 'app' extra. The adapter itself does not need pandas to read CSV bytes."""
    try:
        _import_pandas()
    except ImportError as exc:
        raise MixedError(PANDAS_REQUIRED, "Training on tables with categorical predictors needs the optional 'app' extra "
                         "(pandas). Install the package with [app], or train a numeric-only table.") from exc


def _validate_options(models, split_mode, calibration, fallback_policy, seed, timeout_seconds, review_fractions, costs,
                      description, group_column, time_column):
    if not models or len(set(models)) != len(models) or any(name not in MODEL_NAMES for name in models):
        raise MixedError(OPTION_INVALID, "Choose unique supported model names.")
    if fallback_policy != "strict":
        raise MixedError(OPTION_INVALID, "Only the 'strict' fallback policy is implemented; failed models are reported, not replaced.")
    if split_mode not in SPLIT_MODES:
        raise MixedError(OPTION_INVALID, f"split_mode must be one of {list(SPLIT_MODES)}.")
    if calibration not in CALIBRATIONS:
        raise MixedError(OPTION_INVALID, f"calibration must be one of {list(CALIBRATIONS)}.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MixedError(OPTION_INVALID, "seed must be an integer.")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds)
            or not 1 <= timeout_seconds <= 600):
        raise MixedError(OPTION_INVALID, "timeout_seconds must be between 1 and 600.")
    fractions = list(review_fractions)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in fractions):
        raise MixedError(OPTION_INVALID, "Review fractions must be finite values from 0 to 1.")
    if costs is not None and not isinstance(costs, CostAssumptions):
        raise MixedError(OPTION_INVALID, "costs must be CostAssumptions or None.")
    if description is not None and (not isinstance(description, str) or not description.strip()
                                    or len(description) > MAX_DESCRIPTION_CHARACTERS):
        raise MixedError(OPTION_INVALID, f"description must contain 1..{MAX_DESCRIPTION_CHARACTERS} characters.")
    if split_mode == "group" and group_column is None:
        raise MixedError(OPTION_INVALID, "split_mode='group' needs group_column.")
    if split_mode == "chronological" and time_column is None:
        raise MixedError(OPTION_INVALID, "split_mode='chronological' needs time_column.")
    if split_mode == "chronological" and group_column is not None:
        raise MixedError(SPLIT_UNSUPPORTED, "Combined group and chronological splitting is not implemented; no fallback was used.")
    if split_mode == "chronological" and calibration == "sigmoid":
        raise MixedError(CALIBRATION_INCOMPATIBLE,
                         "Sigmoid calibration with a chronological split is refused: ordered inner folds can split equal "
                         "timestamps. Choose calibration='none' or another split; nothing was downgraded.")
    return fractions


def _notify(progress, event) -> None:
    """Call the optional progress observer. It cannot change or stop a run, so its errors are ignored."""
    if progress is None:
        return
    try:
        progress(event)
    except Exception:  # noqa: BLE001 - an observer must never affect the training result
        pass


def train_comparison_tabular(table_or_csv_bytes, *, target, label_spec, predictors=None, group_column=None, time_column=None,
                             id_columns=(), split_mode="stratified", seed=42, models=DEFAULT_MODELS, calibration="none",
                             limits=TabularLimits(), timeout_seconds=60, missing_policy="error", fallback_policy="strict",
                             review_fractions=(.1, .2), costs=None, description=None, encoding=None, delimiter=None,
                             exclude_columns=(), categorical_handling="auto", require_pandas_for_categorical=False,
                             progress=None):
    """Compare models on a table with numeric, categorical and missing predictors; return a ``training-1.1`` result.

    ``table_or_csv_bytes`` is CSV bytes, a ``ParsedTable``, a DataFrame, a dict of columns or a list of dict rows.
    ``predictors`` selects predictor columns (default: every column that is not the target, group, time or id
    column); unselected columns are listed as excluded with role ``not_selected``. Whole-request problems raise
    typed errors (``TabularError``, ``LabelError`` or ``MixedError``, all ``ValueError`` subclasses with a ``code``)
    before any worker starts; a model that cannot be fitted is reported with ``status: "failed"`` and a
    ``reason_code``. With ``require_pandas_for_categorical`` a table with categorical predictors needs pandas
    (the optional ``app`` extra) and raises ``MixedError(PANDAS_REQUIRED)`` otherwise.

    ``progress`` is an optional callable that receives small JSON-safe dicts: ``{"event": "splitting"}`` before the
    split is built, ``{"event": "model_started", "model": id, "index": i, "total": n}`` and ``{"event":
    "model_finished", "model": id, "index": i, "total": n, "status": "success"|"failed"|"timeout"}`` (plus
    ``"reason_code"`` when the model did not succeed) around each worker (``index`` counts from 1), and
    ``{"event": "reporting"}`` before the report is assembled. The callback only observes: an exception it raises is
    ignored, and with ``progress=None`` (the default) the result is identical to a call that never mentions it.
    """
    if progress is not None and not callable(progress):
        raise MixedError(OPTION_INVALID, "progress must be a callable or None.")
    fractions = _validate_options(models, split_mode, calibration, fallback_policy, seed, timeout_seconds, review_fractions,
                                  costs, description, group_column, time_column)
    if not isinstance(label_spec, LabelSpec):
        raise MixedError(OPTION_INVALID, "label_spec must be a LabelSpec.")
    if not isinstance(limits, TabularLimits):
        raise MixedError(OPTION_INVALID, "limits must be a TabularLimits instance.")

    # 1. Parse (bytes) or normalize, with byte/row/column limits first; then profile with the cardinality/width limits.
    if isinstance(table_or_csv_bytes, (bytes, bytearray)):
        parsed = parse_table_csv(bytes(table_or_csv_bytes), limits, encoding=encoding, delimiter=delimiter)
        columns, total_rows, parsing = parsed.columns, parsed.row_count, parsed.parsing
        parser_name = "parse_table_csv"
    else:
        columns, total_rows, parsing = _as_columns(table_or_csv_bytes, limits)
        parser_name = "ParsedTable" if isinstance(table_or_csv_bytes, ParsedTable) else "table_object"
    if categorical_handling not in ("auto", "reject"):
        raise MixedError(OPTION_INVALID, "categorical_handling must be 'auto' or 'reject'.")
    names = list(columns)
    roles = {name for name in (target, group_column, time_column, *id_columns) if name is not None}
    excluded_by_user = list(exclude_columns)
    unknown_excluded = [name for name in excluded_by_user if name not in columns]
    if unknown_excluded or any(name in roles for name in excluded_by_user):
        raise MixedError(PREDICTOR_UNKNOWN, "exclude_columns must be existing columns that are not the target, group, time or id column.",
                         unknown=[str(name) for name in unknown_excluded])
    if predictors is None:
        selected = [name for name in names if name not in roles and name not in set(excluded_by_user)]
    else:
        predictors = list(predictors)
        unknown = [name for name in predictors if name not in columns]
        if (unknown or len(set(predictors)) != len(predictors) or any(name in roles for name in predictors)
                or any(name in set(excluded_by_user) for name in predictors)):
            raise MixedError(PREDICTOR_UNKNOWN, "predictors must be unique existing columns that are not the target, group, time, "
                             "id or excluded column.", unknown=[str(name) for name in unknown],
                             in_role_columns=[name for name in predictors if name in roles])
        wanted = set(predictors)
        selected = [name for name in names if name in wanted]
    reduced = {name: columns[name] for name in names if name in roles or name in set(selected)}
    profile = profile_table(reduced, target=target, group_column=group_column, time_column=time_column,
                            id_columns=tuple(id_columns), limits=limits)
    profile["parsing"] = parsing
    not_selected = [name for name in names if name not in roles and name not in set(selected)]
    excluded_columns = [dict(record) for record in profile["excluded_columns"]]
    used_by_split = {"group": split_mode == "group", "time": split_mode == "chronological"}
    for record in excluded_columns:
        if record["role"] in used_by_split:
            record["used_by_split"] = used_by_split[record["role"]]
    excluded_columns += [{"column": name, "role": "excluded_by_user", "reason": "excluded by the caller; not used as a predictor"}
                         if name in set(excluded_by_user) else
                         {"column": name, "role": "not_selected", "reason": "not in the selected predictor list; not used as a predictor"}
                         for name in not_selected]
    kinds = {name: _column_kind(profile["columns"][name]) for name in selected}
    has_categorical = any(kind == "categorical" for kind in kinds.values())
    if has_categorical and categorical_handling == "reject":
        raise MixedError(CATEGORICAL_REJECTED, "The table has categorical predictors and categorical_handling='reject' was requested.",
                         categorical_columns=[name for name, kind in kinds.items() if kind == "categorical"])
    if has_categorical and require_pandas_for_categorical:
        _require_pandas()

    # 2. Labels and row exclusions.
    resolution = resolve_labels(columns[target], label_spec, missing_policy=missing_policy)
    mapping = resolution.mapping
    kept = [int(index) for index in resolution.kept_indices]
    y = resolution.y
    positive = mapping["positive_label"]
    negative = mapping["negative_label"] if label_spec.mode == "binary" else "(all other labels)"
    if label_spec.mode != "binary" and negative in {item["label"] for item in mapping["original_labels"]}:
        raise MixedError(OPTION_INVALID, "A real label collides with the one-vs-rest negative display name.")
    used_rows = len(kept)

    # 3. Typed arrays for the retained rows, then the common outer split.
    _notify(progress, {"event": "splitting"})
    specs, arrays = [], {}
    for index, name in enumerate(selected):
        key = f"col_{index}"
        specs.append({"name": name, "kind": kinds[name], "key": key})
        arrays[key] = (_numeric_array(name, columns[name], kept) if kinds[name] == "numeric"
                       else _categorical_array(columns[name], kept))
    groups = times = None
    if split_mode == "group":
        groups = _group_array(group_column, columns[group_column], kept)
    elif split_mode == "feature_group":
        groups = _mixed_feature_groups(arrays, specs)
    elif split_mode == "chronological":
        times = _time_array(time_column, columns[time_column], kept)
    try:
        split = make_split(y, mode=split_mode, seed=seed, groups=groups, times=times)
    except ValueError as exc:
        raise MixedError(SPLIT_INFEASIBLE, f"The requested {split_mode} split is infeasible: {exc}", requested_mode=split_mode,
                         rows=used_rows) from exc
    if calibration == "sigmoid":
        try:
            _inner_splits(np.zeros(len(split.train)), y[split.train], split_mode,
                          groups[split.train] if groups is not None else None, seed)
        except ValueError as exc:
            raise MixedError(CALIBRATION_INFEASIBLE, str(exc), requested_mode=split_mode) from exc
    patterns = (groups if split_mode == "feature_group" else _mixed_feature_groups(arrays, specs))
    diagnostics = _split_diagnostics(patterns, y, split, groups=groups, times=times)

    # 4. One worker per model on the shared outer partitions.
    labels = np.where(y == 1, positive, negative)
    worker_arrays = {"labels": labels.astype(str), "train": split.train, "validation": split.validation, "test": split.test, **arrays}
    if groups is not None:
        worker_arrays["groups"] = groups
    if times is not None:
        worker_arrays["times"] = times
    numeric_only = not has_categorical
    config_base = {
        "adapter": "tabular", "adapter_version": ADAPTER_VERSION, "positive_label": positive, "negative_label": negative,
        "seed": seed, "split_mode": split_mode, "calibration": calibration, "review_fractions": fractions,
        "costs": asdict(costs) if costs is not None else None,
        "description": (description or DEFAULT_DESCRIPTION).rstrip() + DESCRIPTION_SUFFIX,
        "columns": specs, "limits": {"max_rows": limits.max_rows, "max_expanded_features": limits.max_expanded_features,
                                     "max_categorical_cardinality": limits.max_categorical_cardinality},
    }
    results = {}
    for position, name in enumerate(models, start=1):
        _notify(progress, {"event": "model_started", "model": name, "index": position, "total": len(models)})
        config = dict(config_base, model_name=name)
        with tempfile.TemporaryDirectory(prefix="rare-event-worker-", ignore_cleanup_errors=True) as directory:
            directory = Path(directory)
            _write_arrays(directory / "input.npz", worker_arrays)
            (directory / "config.json").write_text(json.dumps(config, allow_nan=False), encoding="utf-8")
            payload = run_worker_directory(directory, name, timeout_seconds)
        row = {key: value for key, value in payload.items()
               if key not in ("test_scores", "test_truth", "test_indices", "test_groups")}
        spec = REGISTRY[name] if numeric_only else get_mixed_spec(name)
        row["model_record"] = spec.to_record(seed)
        if row["status"] == "success":
            row["execution"]["split"] = diagnostics
        else:
            row["reason_code"] = payload.get("error_code") or failure_reason_code(payload)
            if row["status"] == "error":
                row["worker_status"] = "error"
                row["status"] = "failed"
        results[name] = row
        finished = {"event": "model_finished", "model": name, "index": position, "total": len(models),
                    "status": row["status"]}
        if row["status"] != "success":
            finished["reason_code"] = row["reason_code"]
        _notify(progress, finished)

    # 5. Assemble the training-1.1 result.
    _notify(progress, {"event": "reporting"})
    successes = [row for row in results.values() if row["status"] == "success"]
    records = [json.dumps(row["execution"]["preprocessing_record"], sort_keys=True) for row in successes]
    first_record = successes[0]["execution"]["preprocessing_record"] if successes else None
    partitions = {part: {"rows": info["rows"], "positive_rows": info["positive_rows"],
                         "negative_rows": info["rows"] - info["positive_rows"],
                         "fraction_of_rows": info["rows"] / used_rows, "indices_sha256": info["indices_sha256"]}
                  for part, info in diagnostics["partitions"].items()}
    summary = {k: v for k, v in profile.items() if k not in ("parsing", "excluded_columns", "predictors")}
    out = {
        "schema_version": TRAINING_SCHEMA, "outcomes": results,
        "interpretation": "New models under a versioned local protocol; historical ERF results are not reproduced by this run.",
        "label_mapping": mapping, "predictors": selected, "predictor_kinds": kinds, "excluded_columns": excluded_columns,
        "input_parsing": {"parser": parser_name, "details": parsing, "rows_input": total_rows, "rows_used": used_rows,
                          "rows_excluded": total_rows - used_rows, "row_exclusions": mapping["excluded_rows"],
                          "row_exclusions_truncated": mapping["excluded_rows_truncated"]},
        "table_profile": summary,
        "split": {"requested": {"mode": split_mode, "seed": seed,
                                "target_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
                                "fractions_are": "approximate; exact counts are under achieved"},
                  "achieved": {"mode": diagnostics["mode"], "rows": used_rows, "partitions": partitions,
                               "fallback_used": False, "identical_across_models": True,
                               "built_once_in_the_parent": True},
                  "note": "A split that cannot be built raises SPLIT_INFEASIBLE before any model runs; no other split is substituted."},
        "model_records": {name: row["model_record"] for name, row in results.items()},
        "preprocessing": {
            "description": PREPROCESSING_DESCRIPTION if numeric_only else MIXED_PREPROCESSING_DESCRIPTION,
            "fitted_on": "training partition only",
            "input_types": "numeric only" if numeric_only else "numeric and categorical, with missing values",
            "mixed_type_support": True,
            "adapter": {"name": "tabular", "version": ADAPTER_VERSION,
                        "recipe_variant": "numeric" if numeric_only else "mixed",
                        "recipe_rule": "Tables without categorical predictors run the numeric recipes; tables with categorical "
                                       "predictors run the mixed variants (own recipe ids).",
                        "unseen_category_rule": UNSEEN_CATEGORY_RULE,
                        "clustering_representation": None if numeric_only else CLUSTERING_REPRESENTATION,
                        "oversampler": "SMOTE (numeric recipes)" if numeric_only else "SMOTENC on integer-coded categorical columns"},
            "record": first_record, "record_identical_across_models": len(set(records)) <= 1 if records else None},
        "fallback_policy": {"policy": fallback_policy, "fallback_used": False,
                            "description": "A failed model is reported with a reason code. No other recipe or resampler is substituted."},
        "failures": [{"model": name, "status": row["status"], "reason_code": row["reason_code"],
                      "message": row.get("error") or f"Worker limit of {row.get('timeout_seconds')} s reached."}
                     for name, row in results.items() if row["status"] != "success"],
        "seed": seed, "feature_names": selected,
    }
    if parsing and parsing.get("input_sha256"):
        out["input_sha256"] = parsing["input_sha256"]
    return out
