"""Saved-prediction evaluation for the application: a thin wrapper over the evaluation package.

No metric, threshold or report logic lives here. ``rare_event_eval.evaluate_predictions`` computes
everything; this module only converts inputs (cost assumptions given as a mapping), turns
``ValueError`` into a typed ``ServiceError`` and checks that the result is strict JSON.
"""
from __future__ import annotations

from collections.abc import Mapping
import json

from .errors import (APP_EVALUATION_INVALID, APP_INPUT_INVALID, APP_PACKAGE_UNAVAILABLE, APP_REPORT_NOT_JSON_SAFE,
                     ServiceError, translate_errors)


def _reject_constant(token):
    raise ValueError(token)


def _package():
    try:
        import rare_event_eval
    except ImportError as exc:
        raise ServiceError(APP_PACKAGE_UNAVAILABLE,
                           "The evaluation package (rare_event_eval) is not installed.") from exc
    return rare_event_eval


def _costs(value):
    package = _package()
    if value is None or isinstance(value, package.CostAssumptions):
        return value
    if isinstance(value, Mapping):
        allowed = {"false_positive_cost", "false_negative_cost", "review_cost"}
        unknown = sorted(set(value) - allowed)
        if unknown or "false_positive_cost" not in value or "false_negative_cost" not in value:
            raise ServiceError(APP_INPUT_INVALID, "costs needs false_positive_cost and false_negative_cost "
                               "(and optionally review_cost).", details={"unknown_keys": unknown})
        try:
            return package.CostAssumptions(**value)
        except ValueError as exc:
            raise ServiceError(APP_INPUT_INVALID, str(exc)) from None
    raise ServiceError(APP_INPUT_INVALID, "costs must be a mapping, CostAssumptions or None.")


def assert_strict_json(report) -> str:
    """Return the report as JSON text; raise if it holds a NaN or Infinity value.

    The check serializes with ``allow_nan=False`` and parses the text back rejecting the bare constants,
    so a label or note that merely spells the word NaN inside a string is not a false alarm.
    """
    try:
        text = json.dumps(report, allow_nan=False)
        json.loads(text, parse_constant=_reject_constant)
    except (TypeError, ValueError):
        raise ServiceError(APP_REPORT_NOT_JSON_SAFE, "The report holds a value that is not valid JSON.") from None
    return text


def evaluate_saved_predictions(labels, model_scores, *, positive_label, negative_label, score_kind, threshold=None,
                               review_budgets=(), costs=None, calibration_bins=10, evaluation_role="unspecified",
                               data_description=None) -> dict:
    """Evaluate saved predictions with ``rare_event_eval.evaluate_predictions`` (schema 1.0).

    Returns the package report unchanged. Invalid input raises ``ServiceError`` with code
    ``APP_EVALUATION_INVALID`` (the package's message is kept, paths removed). Nothing is fitted.
    """
    package = _package()
    with translate_errors():
        cost_object = _costs(costs)
        try:
            report = package.evaluate_predictions(
                labels, model_scores, positive_label=positive_label, negative_label=negative_label,
                score_kind=score_kind, threshold=threshold, review_budgets=review_budgets, costs=cost_object,
                calibration_bins=calibration_bins, evaluation_role=evaluation_role,
                data_description=data_description)
        except ValueError as exc:
            raise ServiceError(APP_EVALUATION_INVALID, str(exc), layer="package") from None
        assert_strict_json(report)
        return report


def render_report_html(report: dict) -> str:
    """Standalone HTML for an ``evaluate`` (schema 1.0) or ``training-1.x`` report, via the package renderer."""
    package = _package()
    with translate_errors():
        try:
            if isinstance(report, Mapping) and "outcomes" in report:
                from rare_event_eval.reports import training_report
                report = training_report(report)
            return package.render_html(report)
        except ValueError as exc:
            raise ServiceError(APP_EVALUATION_INVALID, str(exc), layer="package") from None
