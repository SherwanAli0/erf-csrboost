"""Explicit binary label handling for training data.

Rules enforced here:

* The positive label is always supplied by the caller. It is never inferred from
  class frequency, label order or label spelling.
* Missing or blank targets are never turned into a class. They raise, or, only
  when ``missing_policy='drop_reported'`` is requested, they are excluded and
  listed with counts.
* Singleton classes and single-class targets raise typed errors. Nothing is
  removed to make a request feasible.
* ``binary`` mode needs exactly two distinct labels. More labels need an explicit
  ``one_vs_rest`` choice, and the original-to-binary mapping is retained.
* Probabilities are never inverted. Internal value 1 always means the explicit
  positive label.

This module uses only NumPy and the standard library. Non-string labels are
compared by their ``str()`` form (so ``1`` and ``"1"`` are the same label, while
``1`` and ``1.0`` are not). Only ``None``, NaN-like values and blank strings count
as missing; strings such as ``"NA"`` are real labels unless the caller says
otherwise.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math

import numpy as np

MODES = ("binary", "one_vs_rest")
MISSING_POLICIES = ("error", "drop_reported")
MAX_LISTED_EXCLUSIONS = 100

LABEL_POSITIVE_REQUIRED = "LABEL_POSITIVE_REQUIRED"
LABEL_SPEC_INVALID = "LABEL_SPEC_INVALID"
LABEL_MISSING_TARGET = "LABEL_MISSING_TARGET"
LABEL_SINGLE_CLASS = "LABEL_SINGLE_CLASS"
LABEL_SINGLETON_CLASS = "LABEL_SINGLETON_CLASS"
LABEL_MULTICLASS_REQUIRES_OVR = "LABEL_MULTICLASS_REQUIRES_OVR"
LABEL_POSITIVE_ABSENT = "LABEL_POSITIVE_ABSENT"
LABEL_NEGATIVE_ABSENT = "LABEL_NEGATIVE_ABSENT"
REASON_CODES = (LABEL_POSITIVE_REQUIRED, LABEL_SPEC_INVALID, LABEL_MISSING_TARGET, LABEL_SINGLE_CLASS,
                LABEL_SINGLETON_CLASS, LABEL_MULTICLASS_REQUIRES_OVR, LABEL_POSITIVE_ABSENT,
                LABEL_NEGATIVE_ABSENT)


class LabelError(ValueError):
    """Typed label failure. ``code`` is a stable reason code; ``details`` is JSON-safe."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details

    def to_record(self) -> dict:
        return {"reason_code": self.code, "message": str(self), "details": self.details}


@dataclass(frozen=True)
class LabelSpec:
    """Caller's explicit label choice.

    ``positive`` is required. In ``binary`` mode ``negative`` is optional and, when
    given, must be the only other label. In ``one_vs_rest`` mode every other label
    is negative and ``negative`` must be None.
    """

    positive: object
    negative: object = None
    mode: str = "binary"


@dataclass(frozen=True)
class LabelResolution:
    """Internal 0/1 labels for the retained rows plus the full record."""

    y: np.ndarray
    kept_indices: np.ndarray
    mapping: dict
    exclusions: list = field(default_factory=list)

    def to_record(self) -> dict:
        """JSON-safe mapping record (exclusions are listed up to a documented cap)."""
        return self.mapping


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    return type(value).__name__ in ("NAType", "NaTType")


def _key(value) -> str:
    return value if isinstance(value, str) else str(value)


def _validate_spec(spec: LabelSpec, missing_policy: str) -> tuple[str, str | None]:
    if not isinstance(spec, LabelSpec):
        raise LabelError(LABEL_SPEC_INVALID, "spec must be a LabelSpec.")
    if spec.mode not in MODES:
        raise LabelError(LABEL_SPEC_INVALID, f"mode must be one of {list(MODES)}.", mode=str(spec.mode))
    if missing_policy not in MISSING_POLICIES:
        raise LabelError(LABEL_SPEC_INVALID, f"missing_policy must be one of {list(MISSING_POLICIES)}.",
                         missing_policy=str(missing_policy))
    if _is_missing(spec.positive):
        raise LabelError(LABEL_POSITIVE_REQUIRED,
                         "An explicit positive label is required. It is never inferred from class frequency or order.")
    positive = _key(spec.positive)
    negative = None if spec.negative is None else _key(spec.negative)
    if negative is not None and _is_missing(spec.negative):
        raise LabelError(LABEL_SPEC_INVALID, "The negative label cannot be blank.")
    if negative is not None and negative == positive:
        raise LabelError(LABEL_SPEC_INVALID, "Positive and negative labels must differ.", label=positive)
    if spec.mode == "one_vs_rest" and negative is not None:
        raise LabelError(LABEL_SPEC_INVALID,
                         "one_vs_rest treats every other label as negative; do not supply a negative label.")
    return positive, negative


def resolve_labels(values, spec: LabelSpec, *, missing_policy: str = "error") -> LabelResolution:
    """Map exact original labels to internal 0/1 values using an explicit choice.

    Returns a ``LabelResolution``. ``y`` covers only the retained rows;
    ``kept_indices`` gives their positions in ``values`` so aligned arrays can be
    sliced. Raises ``LabelError`` with a stable ``code`` otherwise.
    """
    positive, negative = _validate_spec(spec, missing_policy)
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise LabelError(LABEL_SPEC_INVALID, "values must be a one-dimensional sequence of labels.")
    items = list(values.tolist() if isinstance(values, np.ndarray) else values)
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise LabelError(LABEL_SPEC_INVALID, "values must be one-dimensional.")
    total = len(items)

    missing_rows = [index for index, value in enumerate(items) if _is_missing(value)]
    if missing_rows and missing_policy == "error":
        raise LabelError(LABEL_MISSING_TARGET,
                         f"{len(missing_rows)} of {total} target values are missing or blank. Missing targets are never "
                         "converted into a class. Fix the data or request missing_policy='drop_reported'.",
                         missing_count=len(missing_rows), first_rows=missing_rows[:10], row_index_base=0)
    missing_set = set(missing_rows)
    kept_indices = [index for index in range(total) if index not in missing_set]
    keys = [_key(items[index]) for index in kept_indices]
    counts = Counter(keys)
    labels = sorted(counts)

    if positive not in counts:
        raise LabelError(LABEL_POSITIVE_ABSENT, "The chosen positive label does not occur in the target column.",
                         positive_label=positive, observed_labels=labels[:50], observed_label_count=len(labels))
    if len(labels) == 1:
        raise LabelError(LABEL_SINGLE_CLASS, "The target has only one distinct label, so no binary task exists.",
                         label=labels[0], rows=counts[labels[0]])
    if spec.mode == "binary":
        if len(labels) > 2:
            raise LabelError(LABEL_MULTICLASS_REQUIRES_OVR,
                             f"The target has {len(labels)} distinct labels. Binary mode needs exactly two. Choose "
                             "mode='one_vs_rest' with an explicit positive label; no label was merged or removed.",
                             observed_label_count=len(labels), observed_labels=labels[:50])
        other = next(label for label in labels if label != positive)
        if negative is not None and negative != other:
            raise LabelError(LABEL_NEGATIVE_ABSENT,
                             "The chosen negative label is not the other label in the target column.",
                             negative_label=negative, observed_labels=labels)
        negative_labels = [other]
    else:
        negative_labels = [label for label in labels if label != positive]

    positive_rows = counts[positive]
    negative_rows = len(keys) - positive_rows
    singletons = [name for name, rows in (("positive", positive_rows), ("negative", negative_rows)) if rows < 2]
    if singletons:
        raise LabelError(LABEL_SINGLETON_CLASS,
                         f"The {singletons[0]} class has a single row. Singleton classes are never removed; a split "
                         "and a fit cannot use them.", positive_rows=positive_rows, negative_rows=negative_rows)

    y = np.fromiter((1 if key == positive else 0 for key in keys), dtype=np.int64, count=len(keys))
    exclusions = [{"row_index": index, "reason_code": LABEL_MISSING_TARGET,
                   "reason": "Target value is missing or blank; row excluded because missing_policy='drop_reported'."}
                  for index in missing_rows]
    mapping = {
        "mode": spec.mode,
        "positive_label": positive,
        "negative_label": negative_labels[0] if spec.mode == "binary" else None,
        "negative_labels": negative_labels,
        "internal_encoding": {"1": "positive_label", "0": "negative_labels"},
        "original_labels": [{"label": label, "count": counts[label], "internal": 1 if label == positive else 0}
                            for label in labels],
        "counts": {"input_rows": total, "missing_target_rows": len(missing_rows), "effective_rows": len(keys),
                   "positive_rows": positive_rows, "negative_rows": negative_rows},
        "missing_policy": missing_policy,
        "row_index_base": 0,
        "excluded_rows": exclusions[:MAX_LISTED_EXCLUSIONS],
        "excluded_rows_truncated": len(exclusions) > MAX_LISTED_EXCLUSIONS,
        "positive_selection": "explicit; not inferred from class frequency, order or spelling",
        "probability_orientation": "Scores are for internal value 1 (the positive label) and are never inverted.",
    }
    return LabelResolution(y=y, kept_indices=np.asarray(kept_indices, dtype=np.int64), mapping=mapping,
                           exclusions=exclusions)
