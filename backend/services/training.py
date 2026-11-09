"""Training for the application: a thin call into ``rare_event_eval.mixed.train_comparison_tabular``.

The service only prepares the input (see ``tables.py``), converts a cost mapping and translates errors.
The split, preprocessing, resampling, threshold selection and report all come from the package, and the
return value is the package's ``training-1.1`` result unchanged. The caller supplies the target, the
positive label and every exclusion explicitly; nothing is inferred here.
"""
from __future__ import annotations

from .errors import APP_INPUT_INVALID, ServiceError, translate_errors
from .evaluation import _costs, _package, assert_strict_json
from .tables import prepare_input


def app_default_models() -> tuple:
    """Default model ids of the package registry (not the legacy three-way comparison)."""
    from rare_event_eval.registry import default_model_ids
    return default_model_ids()


def train_for_app(data, *, target: str, positive_label, negative_label=None, one_vs_rest: bool = False,
                  filename: str = "", predictors=None, exclude_columns=(), id_columns=(), group_column=None,
                  time_column=None, split_mode: str = "stratified", models=None, seed: int = 42,
                  calibration: str = "none", limits=None, timeout_seconds: float = 60, missing_policy: str = "error",
                  review_fractions=(.1, .2), costs=None, description=None) -> dict:
    """Compare package models on an upload (bytes) or table object; return the ``training-1.1`` result.

    ``positive_label`` is required. Give ``negative_label`` for a binary target, or ``one_vs_rest=True`` for
    more than two labels. ``exclude_columns`` and ``id_columns`` must name the columns to leave out (for
    example the profile's suggested exclusions); a high-cardinality predictor raises ``TABLE_CARDINALITY_EXCEEDED``
    instead of being dropped. A model that cannot be fitted is reported in the result with a reason code; a
    problem with the whole request raises ``ServiceError``.
    """
    with translate_errors():
        _package()
        from rare_event_eval.labels import LabelSpec
        from rare_event_eval.mixed import train_comparison_tabular
        from rare_event_eval.tabular import TabularLimits
        if not isinstance(target, str) or not target:
            raise ServiceError(APP_INPUT_INVALID, "target must be a column name.")
        if one_vs_rest == (negative_label is not None):
            raise ServiceError(APP_INPUT_INVALID, "Give negative_label for a binary target, or one_vs_rest=True "
                               "with no negative_label.")
        limits = limits if limits is not None else TabularLimits()
        prepared = prepare_input(data, filename, limits)
        spec = (LabelSpec(positive_label, mode="one_vs_rest") if one_vs_rest
                else LabelSpec(positive_label, negative_label))
        result = train_comparison_tabular(
            prepared.table, target=target, label_spec=spec, predictors=predictors,
            exclude_columns=tuple(exclude_columns), id_columns=tuple(id_columns), group_column=group_column,
            time_column=time_column, split_mode=split_mode, seed=seed,
            models=tuple(models) if models is not None else app_default_models(), calibration=calibration,
            limits=limits, timeout_seconds=timeout_seconds, missing_policy=missing_policy,
            review_fractions=tuple(review_fractions), costs=_costs(costs), description=description)
        assert_strict_json(result)
        return result
