"""Bounded numeric model comparisons with explicit independent partitions."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .core import CostAssumptions, evaluate_predictions
from .labels import LabelError, LabelSpec, resolve_labels
from .policies import select_threshold
from .registry import REGISTRY, default_model_ids, package_model_ids
from .splits import feature_groups, make_split, split_diagnostics

TRAINING_SCHEMA = "training-1.1"
# Derived from the registry: same ids and order as before the registry existed.
MODEL_NAMES = package_model_ids()
DEFAULT_MODELS = default_model_ids()
PREPROCESSING_DESCRIPTION = ("median imputation and standardization fitted within training pipeline; "
                             "all models receive the same numeric inputs")


def make_model(name, *, seed=42, minority_count=10):
    pre = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
           ("scale", StandardScaler())]
    if name == "logistic":
        estimator = LogisticRegression(max_iter=2000, random_state=seed)
    elif name in ("rf_raw", "rf_weighted", "rf_weighted_1000", "legacy_rf_weighted_200"):
        estimator = RandomForestClassifier(n_estimators={"rf_weighted_1000": 1000, "legacy_rf_weighted_200": 200}.get(name, 100),
                                           class_weight=None if name == "rf_raw" else "balanced",
                                           random_state=seed, n_jobs=1)
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=seed)
    else:
        try:
            from imblearn.ensemble import BalancedRandomForestClassifier
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as SamplingPipeline
            from .estimators import CSRBoostClassifier, ERFClassifier
        except ImportError as exc:
            raise ValueError("This model requires optional training dependencies. Install the package with [training].") from exc
        if name == "balanced_rf":
            estimator = BalancedRandomForestClassifier(n_estimators=100, sampling_strategy="all",
                                                       replacement=True, bootstrap=False, random_state=seed, n_jobs=1)
        elif name == "smote_rf":
            if minority_count < 2:
                raise ValueError("SMOTE requires at least two training rows in the smaller class.")
            return SamplingPipeline(pre + [("resample", SMOTE(random_state=seed, k_neighbors=min(5, minority_count - 1))),
                                           ("model", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1))])
        elif name == "csrboost":
            estimator = CSRBoostClassifier(random_state=seed)
        elif name in ("erf", "erf_no_clustering", "erf_no_smote", "erf_one_bag"):
            estimator = ERFClassifier(random_state=seed, n_bags=1 if name == "erf_one_bag" else 20,
                                       resampling={"erf_no_clustering": "smote", "erf_no_smote": "clustered"}.get(name, "clustered_smote"), n_jobs=1)
        else:
            raise ValueError("Unknown model name.")
    return Pipeline(pre + [("model", estimator)])


def _inner_splits(X, y, split_mode, groups, seed):
    if split_mode in ("group", "feature_group"):
        cv = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y, groups))
    elif split_mode == "chronological":
        splits = list(TimeSeriesSplit(3).split(X))
    else:
        splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
    if any(len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2 for tr, te in splits):
        raise ValueError("Requested training-only calibration folds are infeasible; no fallback was used.")
    return splits


def _peak_rss():
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                     "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32")
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return int(counters.PeakWorkingSetSize)
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    except (ImportError, OSError, AttributeError):
        return None


def _fit_one(X, labels, *, model_name, positive_label, negative_label, split_mode="stratified",
             seed=42, groups=None, times=None, calibration="none", review_fractions=(.1, .2), costs=None):
    """Internal worker payload. Raw scores stay local for paired comparisons."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if labels.ndim != 1 or any(not isinstance(label, str) for label in labels):
        raise ValueError("Training labels must be exact strings, with no missing values.")
    if X.ndim != 2 or len(X) != len(labels) or not 10 <= len(X) <= 20000 or not 1 <= X.shape[1] <= 200:
        raise ValueError("Training requires 10..20000 aligned rows and 1..200 numeric features; no sampling is performed.")
    if np.isinf(X).any() or not np.isin(labels, [positive_label, negative_label]).all() or positive_label == negative_label:
        raise ValueError("Invalid numeric features or explicit binary labels.")
    if model_name not in MODEL_NAMES or calibration not in ("none", "sigmoid"):
        raise ValueError("Unknown model or calibration method.")
    fractions = list(review_fractions)
    if any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in fractions):
        raise ValueError("Review fractions must be finite values from 0 to 1.")
    y = (labels == positive_label).astype(int)
    if split_mode == "feature_group":
        groups = feature_groups(X)
    split = make_split(y, mode=split_mode, seed=seed, groups=groups, times=times)
    Xtr, ytr = X[split.train], y[split.train]
    model = make_model(model_name, seed=seed, minority_count=int(np.bincount(ytr).min()))
    if calibration == "sigmoid":
        train_groups = np.asarray(groups)[split.train] if groups is not None else None
        cv = _inner_splits(Xtr, ytr, split_mode, train_groups, seed)
        model = CalibratedClassifierCV(model, method="sigmoid", cv=cv, n_jobs=1, ensemble=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - started
        validation_scores = model.predict_proba(X[split.validation])[:, list(model.classes_).index(1)]
        selection = select_threshold(y[split.validation], validation_scores, costs=costs)
        test_scores = model.predict_proba(X[split.test])[:, list(model.classes_).index(1)]
    budgets = sorted(set(math.ceil(float(fraction) * len(split.test)) for fraction in fractions))
    report = evaluate_predictions(labels[split.test], {model_name: test_scores}, positive_label=positive_label,
                                  negative_label=negative_label, score_kind="probability", threshold=selection["threshold"],
                                  review_budgets=budgets, costs=costs, evaluation_role="test",
                                  data_description="Fresh model fit; held-out predictions from the declared versioned protocol.")
    report["models"][model_name]["threshold_selection"] = selection
    report["execution_mode"] = "fresh_training"
    report["policies"]["selection"] = "Per-model threshold selected on validation data only; test labels excluded from selection."
    report["checks"].update(training_preprocessing="fitted_on_training_partitions",
                            train_test_overlap="row_indices_disjoint", temporal_or_group_independence="checked_for_requested_split")
    report["checks"]["training_leakage"] = "not_certified: data availability and domain semantics still require review"
    diagnostics = split_diagnostics(X, y, split, groups=groups, times=times)
    events = []
    if calibration == "none" and hasattr(model.named_steps["model"], "resampling_events_"):
        events = model.named_steps["model"].resampling_events_
    metadata = {
        "model": model_name, "seed": seed, "split_mode": split_mode, "calibration": calibration,
        "preprocessing": PREPROCESSING_DESCRIPTION,
        "fit_seconds": fit_seconds, "fit_and_evaluation_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": _peak_rss(), "memory_scope": "worker process peak including interpreter and libraries",
        "threads": 1, "warnings": [str(item.message) for item in captured], "resampling_events": events,
        "split": diagnostics, "threshold_selection": selection,
        "model_parameters": {key: value if isinstance(value, (str, int, float, bool, type(None)))
                             and not (isinstance(value, float) and not math.isfinite(value)) else str(value)
                             for key, value in model.get_params(deep=True).items()},
    }
    return {"status": "success", "report": report, "execution": metadata,
            "test_scores": test_scores.tolist(), "test_truth": y[split.test].tolist(),
            "test_indices": split.test.tolist(),
            "test_groups": np.asarray(groups)[split.test].astype(str).tolist() if groups is not None else None}


def bounded_model_run(X, labels, *, timeout_seconds=60, **configuration):
    """Run one model in a disposable subprocess; terminate only that worker on timeout."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be between 1 and 600.")
    config = dict(configuration)
    groups, times = config.pop("groups", None), config.pop("times", None)
    if isinstance(config.get("costs"), CostAssumptions):
        config["costs"] = asdict(config["costs"])
    arrays = {"X": np.asarray(X, dtype=float), "labels": np.asarray(labels, dtype=str)}
    if groups is not None:
        arrays["groups"] = np.asarray(groups, dtype=str)
    if times is not None:
        arrays["times"] = np.asarray(times, dtype=float)
    with tempfile.TemporaryDirectory(prefix="rare-event-worker-", ignore_cleanup_errors=True) as directory:
        directory = Path(directory)
        np.savez(directory / "input.npz", **arrays)
        (directory / "config.json").write_text(json.dumps(config, allow_nan=False), encoding="utf-8")
        return run_worker_directory(directory, configuration["model_name"], timeout_seconds)


def run_worker_directory(directory, model_name, timeout_seconds):
    """Run the worker on a prepared directory (``input.npz`` and ``config.json``) and return its result.

    One disposable subprocess, thread limits in its environment before any import, terminated
    on timeout. Shared by the numeric path and the mixed-type adapter.
    """
    directory = Path(directory)
    environment = dict(os.environ)
    environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    command = [sys.executable, "-B", "-m", "rare_event_eval.worker", str(directory)]
    started = time.perf_counter()
    with (directory / "stderr.log").open("w", encoding="utf-8") as error_log:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=error_log, env=environment)
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=15)
            return {"status": "timeout", "model": model_name, "timeout_seconds": timeout_seconds,
                    "worker_wall_seconds": time.perf_counter() - started}
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            raise
    result_path = directory / "result.json"
    if not result_path.exists():
        return {"status": "error", "model": model_name,
                "error": "Worker exited without a result.", "exit_code": process.returncode,
                "diagnostic": (directory / "stderr.log").read_text(encoding="utf-8")[-2000:]}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["worker_wall_seconds"] = time.perf_counter() - started
    return result


FAILURE_RULES = (
    ("MODEL_UNKNOWN", r"Unknown model name"),
    ("MODEL_DEPENDENCY_MISSING", r"optional training dependencies"),
    ("CALIBRATION_INFEASIBLE", r"calibration folds are infeasible"),
    ("RESAMPLING_INFEASIBLE", r"SMOTE requires"),
    ("SPLIT_INFEASIBLE", r"[Ss]plit|[Gg]roup|Every partition|[Tt]ime values|Splitting requires"),
    ("INPUT_INVALID", r"Invalid numeric features|Training labels must|Training requires|Unknown model or calibration|Review fractions"),
)


def failure_reason_code(payload):
    """Stable reason code for a model outcome that is not a success.

    The worker reports an exception type and message; this classifies the message.
    Codes: MODEL_TIMEOUT, WORKER_NO_RESULT, MODEL_UNKNOWN, MODEL_DEPENDENCY_MISSING,
    CALIBRATION_INFEASIBLE, RESAMPLING_INFEASIBLE, SPLIT_INFEASIBLE, INPUT_INVALID,
    MODEL_OUT_OF_MEMORY and MODEL_FIT_ERROR (anything else). The message is always kept.
    """
    if payload.get("status") == "timeout":
        return "MODEL_TIMEOUT"
    message = str(payload.get("error", ""))
    if payload.get("error_type") == "MemoryError":
        return "MODEL_OUT_OF_MEMORY"
    if message.startswith("Worker exited without a result"):
        return "WORKER_NO_RESULT"
    for code, pattern in FAILURE_RULES:
        if re.search(pattern, message):
            return code
    return "MODEL_FIT_ERROR"


def _split_summary(configuration, outcomes, total_rows):
    mode = configuration.get("split_mode", "stratified")
    requested = {"mode": mode, "seed": configuration.get("seed", 42),
                 "target_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
                 "fractions_are": "approximate; exact counts are under achieved"}
    successes = [row["execution"]["split"] for row in outcomes.values() if row.get("status") == "success"]
    if not successes:
        return {"requested": requested, "achieved": None,
                "note": "No model completed, so no achieved split is recorded. See failures for reason codes."}
    first = successes[0]
    partitions = {name: {"rows": info["rows"], "positive_rows": info["positive_rows"],
                         "negative_rows": info["rows"] - info["positive_rows"],
                         "fraction_of_rows": info["rows"] / total_rows, "indices_sha256": info["indices_sha256"]}
                  for name, info in first["partitions"].items()}
    return {"requested": requested,
            "achieved": {"mode": first["mode"], "rows": total_rows, "partitions": partitions,
                         "fallback_used": False,
                         "identical_across_models": all(item["partitions"] == first["partitions"] for item in successes)},
            "note": "A split that cannot be built fails with SPLIT_INFEASIBLE; no other split is substituted."}


def train_comparison(X, labels, *, models=DEFAULT_MODELS, label_spec=None, missing_policy="error",
                     fallback_policy="strict", report_context=None, **configuration):
    """Public training API: return aggregate reports and explicit model outcomes.

    Labels are resolved explicitly before any worker starts (see ``labels.py``); a
    ``LabelError`` is raised, not reported per model, if they cannot be. Pass
    ``label_spec`` for one-versus-rest or to choose ``missing_policy='drop_reported'``;
    otherwise ``positive_label`` and ``negative_label`` in the configuration define a
    binary spec. ``report_context`` may carry ``feature_names``, ``excluded_columns``,
    ``parsing``, ``parser`` and ``input_sha256`` for the report. Only the ``strict``
    fallback policy exists: a model that cannot be built or fitted is reported as a
    failure with a reason code, never replaced by another recipe.
    """
    if not models or len(set(models)) != len(models) or any(name not in MODEL_NAMES for name in models):
        raise ValueError("Choose unique supported model names.")
    if fallback_policy != "strict":
        raise ValueError("Only the 'strict' fallback policy is implemented; failed models are reported, not replaced.")
    context = dict(report_context or {})
    if label_spec is None:
        label_spec = LabelSpec(configuration.get("positive_label"), configuration.get("negative_label"))
    elif any(configuration.get(key) not in (None, value) for key, value in
             (("positive_label", label_spec.positive), ("negative_label", label_spec.negative))):
        raise ValueError("positive_label/negative_label conflict with label_spec.")
    resolution = resolve_labels(labels, label_spec, missing_policy=missing_policy)
    mapping = resolution.mapping
    total_input = mapping["counts"]["input_rows"]
    used_rows = int(len(resolution.kept_indices))
    configuration["positive_label"] = mapping["positive_label"]
    if label_spec.mode == "binary":
        configuration["negative_label"] = mapping["negative_label"]
    else:
        configuration["negative_label"] = "(all other labels)"
        if configuration["negative_label"] in {item["label"] for item in mapping["original_labels"]}:
            raise LabelError("LABEL_SPEC_INVALID", "A real label collides with the one-vs-rest negative display name.")
    if used_rows != total_input or label_spec.mode == "one_vs_rest":
        X = np.asarray(X, dtype=float)
        if len(X) != total_input:
            raise ValueError("X and labels must have the same number of rows.")
        kept = resolution.kept_indices
        X = X[kept]
        for key in ("groups", "times"):
            if configuration.get(key) is not None:
                configuration[key] = np.asarray(configuration[key])[kept]
        labels = np.where(resolution.y == 1, configuration["positive_label"], configuration["negative_label"])
    seed = configuration.get("seed", 42)
    results = {}
    for name in models:
        payload = bounded_model_run(X, labels, model_name=name, **configuration)
        row = {key: value for key, value in payload.items()
               if key not in ("test_scores", "test_truth", "test_indices", "test_groups")}
        row["model_record"] = REGISTRY[name].to_record(seed)
        if row["status"] != "success":
            row["reason_code"] = failure_reason_code(payload)
        results[name] = row
    out = {"schema_version": TRAINING_SCHEMA, "outcomes": results,
           "interpretation": "New numeric models under a versioned local protocol; historical ERF results are not reproduced by this run.",
           "label_mapping": mapping,
           "predictors": context.get("feature_names"),
           "excluded_columns": context.get("excluded_columns", []),
           "input_parsing": {"parser": context.get("parser", "caller_supplied_arrays"), "details": context.get("parsing"),
                             "rows_input": total_input, "rows_used": used_rows, "rows_excluded": total_input - used_rows,
                             "row_exclusions": mapping["excluded_rows"],
                             "row_exclusions_truncated": mapping["excluded_rows_truncated"]},
           "split": _split_summary(configuration, results, used_rows),
           "model_records": {name: row["model_record"] for name, row in results.items()},
           "preprocessing": {"description": PREPROCESSING_DESCRIPTION, "fitted_on": "training partition only",
                             "input_types": "numeric only; categorical input is not supported by these recipes",
                             "mixed_type_support": False},
           "fallback_policy": {"policy": fallback_policy, "fallback_used": False,
                               "description": "A failed model is reported with a reason code. No other recipe or resampler is substituted."},
           "failures": [{"model": name, "status": row["status"], "reason_code": row["reason_code"],
                         "message": row.get("error") or f"Worker limit of {row.get('timeout_seconds')} s reached."}
                        for name, row in results.items() if row["status"] != "success"],
           "seed": seed}
    if context.get("input_sha256") is not None:
        out["input_sha256"] = context["input_sha256"]
    if context.get("feature_names") is not None:
        out["feature_names"] = context["feature_names"]
    return out
