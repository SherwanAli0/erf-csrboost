"""Binary prediction evaluation. All policies are fixed before this call.

Input order breaks score ties. Bounds at a tied top-k cutoff describe possible
tie allocations, not statistical confidence intervals. No training code runs here.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from numbers import Integral, Real
import math
import platform

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from . import __version__


def _number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative." if nonnegative else "."))
    return result


@dataclass(frozen=True)
class CostAssumptions:
    """Hypothetical additive costs in one user-defined unit.

    Every flagged row incurs review_cost. Incorrect flags also incur
    false_positive_cost. Unflagged positives incur false_negative_cost.
    These assumptions do not measure business savings or reviewer effectiveness.
    """

    false_positive_cost: float
    false_negative_cost: float
    review_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in ("false_positive_cost", "false_negative_cost", "review_cost"):
            object.__setattr__(self, name, _number(getattr(self, name), name, nonnegative=True))

    def as_dict(self) -> dict:
        return {
            "false_positive_cost": self.false_positive_cost,
            "false_negative_cost": self.false_negative_cost,
            "review_cost": self.review_cost,
            "unit": "user-defined cost unit",
            "formula": "FP * false_positive_cost + FN * false_negative_cost + flagged * review_cost",
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _decision(y: np.ndarray, selected: np.ndarray, costs: CostAssumptions | None) -> dict:
    positive = y == 1
    tp = int(np.count_nonzero(positive & selected))
    fp = int(np.count_nonzero(~positive & selected))
    fn = int(np.count_nonzero(positive & ~selected))
    tn = int(np.count_nonzero(~positive & ~selected))
    result = {
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "flagged": tp + fp, "flagged_fraction": float((tp + fp) / len(y)),
        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp), "false_positive_rate": _ratio(fp, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn), "accuracy": float((tp + tn) / len(y)),
        "assumed_total_cost": None,
    }
    recall, specificity = result["recall"], result["specificity"]
    result["balanced_accuracy"] = ((recall + specificity) / 2
                                   if recall is not None and specificity is not None else None)
    if costs is not None:
        amount = (fp * costs.false_positive_cost + fn * costs.false_negative_cost
                  + (tp + fp) * costs.review_cost)
        if not math.isfinite(amount):
            raise ValueError("Calculated costs exceed the finite numeric range.")
        result["assumed_total_cost"] = float(amount)
    return result


def _at_budget(y: np.ndarray, scores: np.ndarray, order: np.ndarray, budget: int,
               costs: CostAssumptions | None) -> dict:
    selected = np.zeros(len(y), dtype=bool)
    selected[order[:budget]] = True
    result = {"budget": budget, **_decision(y, selected, costs), "boundary_tie": None}
    if 0 < budget < len(y) and scores[order[budget - 1]] == scores[order[budget]]:
        boundary = scores[order[budget - 1]]
        strictly_above = scores > boundary
        tied = scores == boundary
        slots = budget - int(strictly_above.sum())
        positives_above = int(y[strictly_above].sum())
        tied_positives = int(y[tied].sum())
        tied_negatives = int(tied.sum()) - tied_positives
        low = positives_above + max(0, slots - tied_negatives)
        high = positives_above + min(slots, tied_positives)
        result["boundary_tie"] = {
            "score": float(boundary), "tied_rows": int(tied.sum()), "available_slots": slots,
            "true_positives_min": low, "true_positives_max": high,
            "precision_min": float(low / budget), "precision_max": float(high / budget),
            "recall_min": _ratio(low, int(y.sum())), "recall_max": _ratio(high, int(y.sum())),
            "interpretation": "Tie-allocation bounds, not confidence intervals.",
        }
    return result


def _calibration(y: np.ndarray, scores: np.ndarray, bins: int) -> dict:
    positions = np.minimum((scores * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        mask = positions == index
        count = int(mask.sum())
        rows.append({
            "lower": index / bins, "upper": (index + 1) / bins,
            "upper_inclusive": index == bins - 1, "count": count,
            "mean_score": float(scores[mask].mean()) if count else None,
            "observed_positive_fraction": float(y[mask].mean()) if count else None,
        })
    return {
        "strategy": "equal_width", "bins": rows,
        "interpretation": "Descriptive reliability bins; not evidence that probabilities are calibrated.",
    }


def evaluate_predictions(
    labels: Iterable[str],
    model_scores: Mapping[str, Iterable[float]],
    *,
    positive_label: str,
    negative_label: str,
    score_kind: str,
    threshold: float | None = None,
    review_budgets: Iterable[int] = (),
    costs: CostAssumptions | None = None,
    calibration_bins: int = 10,
    evaluation_role: str = "unspecified",
    data_description: str | None = None,
) -> dict:
    """Evaluate one or more score columns aligned to the same labelled rows.

    Labels are exact strings; nothing is trimmed, merged, dropped or inferred.
    Higher scores must indicate the declared positive class. ``probability``
    scores must lie in [0, 1]; ``ranking`` scores need only be finite.
    Probability scores default to a fixed threshold of 0.5. Ranking scores have
    no threshold unless supplied. Neither thresholds nor budgets are optimized.
    """
    for name, label in (("positive_label", positive_label), ("negative_label", negative_label)):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{name} must be a nonempty string.")
    if positive_label == negative_label:
        raise ValueError("Positive and negative labels must differ.")
    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a sequence of exact string labels.")
    raw_labels = list(labels)
    if not raw_labels:
        raise ValueError("At least one labelled row is required.")
    for index, label in enumerate(raw_labels, start=1):
        if not isinstance(label, str) or label not in (positive_label, negative_label):
            raise ValueError(f"Unknown or missing label at row {index}; no rows were dropped.")
    y = np.asarray([int(label == positive_label) for label in raw_labels], dtype=np.int8)
    n = len(y)
    if score_kind not in ("probability", "ranking"):
        raise ValueError("score_kind must be 'probability' or 'ranking'.")
    if evaluation_role not in ("test", "validation", "unspecified"):
        raise ValueError("evaluation_role must be 'test', 'validation' or 'unspecified'.")
    if data_description is not None and (not isinstance(data_description, str)
                                         or not data_description.strip() or len(data_description) > 500):
        raise ValueError("data_description must be a nonempty string of at most 500 characters, or None.")
    if costs is not None and not isinstance(costs, CostAssumptions):
        raise ValueError("costs must be CostAssumptions or None.")
    if (isinstance(calibration_bins, (bool, np.bool_)) or not isinstance(calibration_bins, Integral)
            or not 2 <= calibration_bins <= 50):
        raise ValueError("calibration_bins must be an integer from 2 to 50.")
    budgets = []
    for budget in review_budgets:
        if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Integral):
            raise ValueError("Review budgets must be integer row counts.")
        if not 0 <= budget <= n:
            raise ValueError(f"Every review budget must be between 0 and {n}; no capping is performed.")
        if budget in budgets:
            raise ValueError("Review budgets must be unique.")
        budgets.append(int(budget))
    if threshold is None and score_kind == "probability":
        threshold = 0.5
    if threshold is not None:
        threshold = _number(threshold, "threshold")
        if score_kind == "probability" and not 0 <= threshold <= 1:
            raise ValueError("A probability threshold must lie between 0 and 1.")
    if not isinstance(model_scores, Mapping) or not model_scores:
        raise ValueError("model_scores must be a nonempty mapping of model names to score sequences.")
    positive_count = int(y.sum())
    both_classes = 0 < positive_count < n
    models = {}
    for name, values in model_scores.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each model name must be a nonempty string.")
        if isinstance(values, (str, bytes)):
            raise ValueError("Scores must be a one-dimensional sequence of numbers.")
        try:
            scores = np.asarray(list(values), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Each score must be numeric; no rows were dropped.") from exc
        if scores.ndim != 1 or len(scores) != n:
            raise ValueError("Every model must provide one score for each labelled row.")
        if not np.isfinite(scores).all():
            raise ValueError("Scores must be finite; NaN and infinity are rejected.")
        if score_kind == "probability" and ((scores < 0).any() or (scores > 1).any()):
            raise ValueError("Probability scores must lie between 0 and 1.")
        order = np.argsort(-scores, kind="stable")
        budget_results = [_at_budget(y, scores, order, budget, costs) for budget in budgets]
        warnings = []
        if not both_classes:
            warnings.append("Only one class is present. AP and ROC AUC are unavailable in this report.")
        if positive_count < 20:
            warnings.append("Fewer than 20 positive rows: estimates can be unstable; this is a descriptive warning, not a validity cutoff.")
        if np.unique(scores).size <= 2:
            warnings.append("Only one or two distinct scores are present; ranking and budget results may depend strongly on ties.")
        if any(row["boundary_tie"] is not None for row in budget_results):
            warnings.append("A review budget cuts through a score tie. Input row order breaks ties; inspect the allocation bounds.")
        metrics = {
            "average_precision": float(average_precision_score(y, scores)) if both_classes else None,
            "roc_auc": float(roc_auc_score(y, scores)) if both_classes else None,
            "brier_score": None, "log_loss": None,
        }
        if score_kind == "probability":
            metrics["brier_score"] = float(brier_score_loss(y, scores, pos_label=1))
            metrics["log_loss"] = float(log_loss(y, np.column_stack([1 - scores, scores]), labels=[0, 1]))
        models[name] = {
            "metrics": metrics,
            "threshold_result": ({"threshold": threshold, **_decision(y, scores >= threshold, costs)}
                                 if threshold is not None else None),
            "review_budget_results": budget_results,
            "calibration": _calibration(y, scores, int(calibration_bins)) if score_kind == "probability" else None,
            "warnings": warnings,
        }
    return {
        "schema_version": "1.0",
        "tool": {"name": "rare-event-eval", "version": __version__, "python": platform.python_version(),
                 "numpy": np.__version__, "scikit_learn": version("scikit-learn")},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": {"kind": "binary_classification", "positive_label": positive_label,
                 "negative_label": negative_label, "score_kind": score_kind,
                 "evaluation_role": evaluation_role, "role_verified": False},
        "data": {"rows": n, "positive_rows": positive_count, "negative_rows": n - positive_count,
                 "positive_prevalence": positive_count / n, "rows_dropped": 0,
                 "description": data_description, "description_verified": False},
        "policies": {"threshold": threshold, "threshold_rule": "score >= threshold",
                     "selection": "Fixed policies supplied before this evaluation; no optimization performed.",
                     "review_budgets": budgets, "tie_break": "input row order",
                     "cost_assumptions": costs.as_dict() if costs else None},
        "checks": {"labels_and_scores": "validated", "training_leakage": "not_checked",
                   "train_test_overlap": "not_checked", "temporal_or_group_independence": "not_checked",
                   "uncertainty": "not_estimated"},
        "models": models,
        "limitations": [
            "Saved predictions cannot establish that model training, splitting or preprocessing avoided leakage.",
            "The evaluation role is user-declared. Score orientation and row alignment must be supplied correctly.",
            "Inspecting many policies on a test set is exploratory; choose a final policy using independent validation data.",
            "Metrics are descriptive point estimates, not confidence intervals or evidence of general superiority.",
            "Calibration bins are descriptive. A probability score is not automatically a calibrated event probability.",
            "Average precision uses recall-increment weighting, not trapezoidal PR area; prevalence affects its interpretation.",
            "Cost estimates use hypothetical additive assumptions and do not establish realized savings or review outcomes.",
            "JSON and HTML contain aggregates, not raw labels or prediction rows; aggregates can still be sensitive.",
        ],
    }
