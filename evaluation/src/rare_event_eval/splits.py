"""Explicit, disjoint train/validation/test construction and diagnostics."""
from __future__ import annotations

from hashlib import sha256
import json
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def feature_groups(X):
    """Exact feature-pattern groups, including a canonical representation of NaN.

    No targets are used. Grouping defines generalization to new feature patterns;
    repeated feature patterns are not automatically a proven leakage defect.
    """
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array.")
    keys = [sha256(json.dumps([None if np.isnan(v) else (0.0 if v == 0 else float(v)) for v in row],
                             separators=(",", ":")).encode()).hexdigest() for row in values]
    return np.asarray(keys)


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    mode: str


def make_split(y, *, mode="stratified", seed=42, groups=None, times=None):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) < 10 or len(np.unique(y)) != 2:
        raise ValueError("Splitting requires two classes and at least ten rows.")
    indices = np.arange(len(y))
    if mode == "stratified":
        trainval, test = train_test_split(indices, test_size=.2, stratify=y, random_state=seed)
        train, validation = train_test_split(trainval, test_size=.25, stratify=y[trainval], random_state=seed)
    elif mode in ("group", "feature_group"):
        if groups is None or len(groups) != len(y):
            raise ValueError("Group splitting requires one nonmissing group per row.")
        groups = np.asarray(groups)
        if groups.ndim != 1 or any(value is None or str(value).strip() == "" or (isinstance(value, float) and np.isnan(value)) for value in groups):
            raise ValueError("Group identifiers must be a one-dimensional nonmissing array.")
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        trainval, test = next(outer.split(indices, y, groups))
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
        tr, va = next(inner.split(trainval, y[trainval], groups[trainval]))
        train, validation = trainval[tr], trainval[va]
    elif mode == "chronological":
        if times is None or len(times) != len(y):
            raise ValueError("Chronological splitting requires one comparable time per row.")
        times = np.asarray(times)
        if times.ndim != 1 or any(value is None or str(value).strip() in ("", "NaT", "nan") for value in times):
            raise ValueError("Time values cannot be missing.")
        if not (np.issubdtype(times.dtype, np.number) or np.issubdtype(times.dtype, np.datetime64)):
            raise ValueError("Time values must be numeric or datetime64, not lexically sorted strings.")
        if np.issubdtype(times.dtype, np.number) and not np.isfinite(times).all():
            raise ValueError("Time values must be finite.")
        order = np.argsort(times, kind="stable")
        # Whole timestamp groups stay together. Fractions are approximate.
        first, second = int(len(y) * .6), int(len(y) * .8)
        while first < len(y) and times[order[first]] == times[order[first - 1]]:
            first += 1
        second = max(second, first + 1)
        while second < len(y) and times[order[second]] == times[order[second - 1]]:
            second += 1
        train, validation, test = order[:first], order[first:second], order[second:]
        if groups is not None:
            raise ValueError("Combined group and chronological splitting is not implemented; no fallback was used.")
    else:
        raise ValueError("Unknown split mode.")
    partitions = [np.asarray(part, dtype=int) for part in (train, validation, test)]
    if any(len(part) == 0 or len(np.unique(y[part])) != 2 for part in partitions):
        raise ValueError("Every partition must contain both classes. Requested split is infeasible; no fallback was used.")
    if len(set(np.concatenate(partitions).tolist())) != len(y):
        raise ValueError("Split indices are not a disjoint partition of the input.")
    if mode in ("group", "feature_group"):
        sets = [set(groups[part].tolist()) for part in partitions]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Group overlap detected.")
    return Split(*partitions, mode)


def split_diagnostics(X, y, split, *, groups=None, times=None):
    patterns = feature_groups(X)
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
                               "indices_sha256": sha256(np.sort(part).astype('<i8').tobytes()).hexdigest()}
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
