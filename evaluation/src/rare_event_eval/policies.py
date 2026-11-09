"""Validation-only decision selection and conditional paired uncertainty."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from .core import CostAssumptions


def select_threshold(y_validation, scores_validation, *, costs: CostAssumptions | None = None):
    y = np.asarray(y_validation)
    scores = np.asarray(scores_validation, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires aligned binary validation labels and scores.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Threshold selection requires 0/1 labels and finite probabilities.")
    candidates = []
    for threshold in np.linspace(0, 1, 101):
        selected = scores >= threshold
        tp = int(((y == 1) & selected).sum())
        fp = int(((y == 0) & selected).sum())
        fn = int(((y == 1) & ~selected).sum())
        if costs is None:
            value = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
            objective = value
        else:
            value = fp * costs.false_positive_cost + fn * costs.false_negative_cost + int(selected.sum()) * costs.review_cost
            objective = -value
        candidates.append((objective, float(threshold), float(value)))
    _, threshold, value = max(candidates)  # deterministic highest-threshold tie break
    return {"threshold": threshold, "source": "validation_only", "objective": "minimize_assumed_cost" if costs else "maximize_f1",
            "validation_objective_value": value, "validation_rows": len(y),
            "search": "101 fixed thresholds from 0 to 1 inclusive; equality flags the row",
            "tie_break": "highest threshold among equal objectives", "test_labels_used": False}


def paired_ap_interval(y, first, second, *, groups=None, repetitions=200, seed=42, chronological=False):
    """Percentile interval for AP(first)-AP(second), conditional on fitted models.

    Resample identical row/group draws for both models. This is exploratory and
    does not account for training variation, selection or multiple comparisons.
    No IID interval is substituted for a chronological evaluation.
    """
    if chronological:
        return {"available": False, "reason": "Temporal dependence needs an explicitly justified time/block resampling design."}
    y, first, second = np.asarray(y), np.asarray(first), np.asarray(second)
    if (y.ndim != 1 or first.shape != y.shape or second.shape != y.shape
            or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("Paired bootstrap requires aligned binary labels and finite scores for both models.")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 20:
        raise ValueError("At least 20 bootstrap repetitions are required.")
    units = np.arange(len(y)) if groups is None else np.asarray(groups)
    if units.ndim != 1 or len(units) != len(y):
        raise ValueError("One group is required per evaluation row.")
    _, inverse = np.unique(units, return_inverse=True)
    members = [np.flatnonzero(inverse == group) for group in range(int(inverse.max()) + 1)]
    rng, differences = np.random.default_rng(seed), []
    for _ in range(repetitions):
        sample = np.concatenate([members[index] for index in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[sample])) < 2:
            continue
        differences.append(float(average_precision_score(y[sample], first[sample])
                                 - average_precision_score(y[sample], second[sample])))
    if len(differences) < max(20, repetitions // 2):
        return {"available": False, "reason": "Too few bootstrap samples contain both classes.",
                "requested_repetitions": repetitions, "valid_repetitions": len(differences)}
    return {"available": True, "metric": "AP(first) - AP(second)",
            "estimate": float(average_precision_score(y, first) - average_precision_score(y, second)),
            "lower": float(np.quantile(differences, .025)), "upper": float(np.quantile(differences, .975)),
            "confidence_level": .95, "method": "paired group percentile bootstrap" if groups is not None else "paired row percentile bootstrap",
            "requested_repetitions": repetitions, "valid_repetitions": len(differences), "seed": seed,
            "interpretation": "Exploratory interval conditional on fitted models and exchangeable sampling units; excludes training variation and multiplicity."}
