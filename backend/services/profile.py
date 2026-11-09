"""Dataset profile for the application, built from the package profile and the legacy target suggestion.

``profile_for_app`` never decides anything for the user. The target suggestion comes from the legacy
``engine.profile`` and is marked ``suggested, needs confirmation``. The positive-class suggestion
(the rarest label by count, which is what the legacy app silently assumed) is marked the same way.
Columns the legacy app dropped silently as identifier-like or free text are listed as suggested
exclusions; the caller must confirm them, because the package never removes a column on its own.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from .errors import APP_TARGET_REQUIRED, ServiceError, translate_errors
from .evaluation import _package
from .tables import PreparedInput, legacy_engine, prepare_input

SCHEMA = "app-profile-1.0"
NEEDS_CONFIRMATION = "suggested, needs confirmation"
MAX_LISTED_LABELS = 50


def _limits(limits):
    from rare_event_eval.tabular import TabularLimits
    return limits if limits is not None else TabularLimits()


def _limit_status(prepared: PreparedInput, *, target, group_column, time_column, id_columns, limits) -> dict:
    """Run the package profile; report violations as data instead of raising."""
    from rare_event_eval.tabular import TabularError, profile_table
    try:
        record = profile_table(prepared.table, target=target, group_column=group_column, time_column=time_column,
                               id_columns=tuple(id_columns), limits=limits)
    except TabularError as exc:
        violations = exc.details.get("violations") if isinstance(exc.details, dict) else None
        return {"status": "exceeded" if violations or str(exc.code).startswith("TABLE_TOO") else "invalid",
                "code": exc.code, "message": str(exc)[:300], "violations": violations or [], "profile": None}
    return {"status": "ok", "code": None, "message": None, "violations": [], "profile": record}


def _summary(record: dict | None) -> dict | None:
    if record is None:
        return None
    keys = ("rows", "columns_total", "predictor_count", "predictors", "constant_columns", "empty_columns",
            "duplicate_columns", "duplicate_feature_groups", "expanded_feature_estimate", "target_summary")
    out = {key: record[key] for key in keys}
    out["columns"] = {name: {key: info[key] for key in ("role", "kind", "value_type", "distinct", "distinct_capped",
                                                         "missing", "missing_fraction", "expanded_width")}
                      for name, info in record["columns"].items()}
    return out


def _target_counts(values) -> tuple[Counter, int]:
    from rare_event_eval.labels import _is_missing
    present = [item if isinstance(item, str) else str(item) for item in values if not _is_missing(item)]
    return Counter(present), len(values) - len(present)


def _class_summary(prepared: PreparedInput, target: str) -> dict:
    counts, missing = _target_counts(prepared.columns[target])
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    out = {"column": target, "missing_target_rows": missing, "distinct_labels": len(counts),
           "labels": [{"label": label, "count": count} for label, count in ordered[:MAX_LISTED_LABELS]],
           "labels_truncated": len(ordered) > MAX_LISTED_LABELS, "suggested_positive": None,
           "suggested_label_mode": None}
    if len(counts) >= 2:
        rarest = min(counts.items(), key=lambda item: (item[1], item[0]))[0]
        out["suggested_positive"] = {"label": rarest, "status": NEEDS_CONFIRMATION,
                                     "basis": "rarest label by count (the rule the legacy app applied without asking)"}
        out["suggested_label_mode"] = "binary" if len(counts) == 2 else "one_vs_rest"
    return out


def _suggestion(prepared: PreparedInput) -> tuple[dict | None, list]:
    """Target suggestion and legacy schema from ``engine.profile`` (needs pandas)."""
    if prepared.frame is None:
        return None, []
    engine = legacy_engine()
    legacy = engine.profile(prepared.frame)
    return ({"column": legacy["suggested_target"], "status": NEEDS_CONFIRMATION,
             "confident": bool(legacy["target_confident"]), "reason": legacy["guess_reason"] or None,
             "source": "legacy engine.profile"}, legacy["columns"])


def profile_for_app(data, filename: str = "", *, target: str | None = None, group_column: str | None = None,
                    time_column: str | None = None, id_columns=(), limits=None) -> dict:
    """Profile an upload (bytes) or a table object for the application.

    Returns a JSON-safe record (schema ``app-profile-1.0``): parsing record, per-column schema, a target
    suggestion flagged ``suggested, needs confirmation``, class counts for the chosen target with a
    positive-class suggestion, excluded columns (roles you set plus suggested exclusions) and the limit status.
    ``target`` defaults to the suggestion when one exists (uploads and DataFrames need pandas); otherwise it
    is required. Violations of a limit are reported under ``limits.status`` and do not raise.
    """
    with translate_errors():
        _package()
        limits = _limits(limits)
        prepared = prepare_input(data, filename, limits)
        suggestion, legacy_columns = _suggestion(prepared)
        chosen = target if target is not None else (suggestion or {}).get("column")
        if chosen is None:
            raise ServiceError(APP_TARGET_REQUIRED, "A target column is required: no suggestion is available for "
                               "this input.")
        if chosen not in prepared.columns:
            from rare_event_eval.tabular import TabularError, TABLE_COLUMN_MISSING
            raise TabularError(TABLE_COLUMN_MISSING, f"The target column {chosen!r} is not in the table.",
                               role="target", column=str(chosen))
        roles = {name for name in (chosen, group_column, time_column, *id_columns) if name is not None}
        suggested_exclusions = [
            {"column": item["column"], "role": "suggested_exclusion", "status": NEEDS_CONFIRMATION,
             "reason": "identifier-like or free text: more than half of the rows hold distinct values "
                       "(the legacy app dropped such columns without asking)"}
            for item in legacy_columns if item.get("id_like") and item["column"] not in roles]
        status = _limit_status(prepared, target=chosen, group_column=group_column, time_column=time_column,
                               id_columns=id_columns, limits=limits)
        after = None
        if suggested_exclusions and status["status"] != "ok":
            ids = tuple(id_columns) + tuple(item["column"] for item in suggested_exclusions)
            after = _limit_status(prepared, target=chosen, group_column=group_column, time_column=time_column,
                                  id_columns=ids, limits=limits)
        record = status["profile"]
        schema = legacy_columns or [{"column": name, "dtype": info["value_type"], "unique": info["distinct"],
                                     "missing": info["missing"], "id_like": None}
                                    for name, info in (record or {}).get("columns", {}).items()]
        return {
            "schema": SCHEMA,
            "rows": prepared.rows, "columns": len(prepared.columns),
            "parsing": prepared.parsing,
            "schema_columns": schema,
            "target": {"column": chosen, "source": "caller" if target is not None else "suggestion",
                       "suggestion": suggestion if suggestion is not None else
                       {"column": None, "status": "not available",
                        "reason": "suggestions need an upload or DataFrame and pandas"}},
            "classes": _class_summary(prepared, chosen),
            "excluded_columns": [{"column": name, "role": role} for role, name in
                                 (("target", chosen), ("group", group_column), ("time", time_column))
                                 if name is not None] + [{"column": name, "role": "id"} for name in id_columns],
            "suggested_exclusions": suggested_exclusions,
            "limits": {"configured": asdict(limits), "status": status["status"], "code": status["code"],
                       "message": status["message"], "violations": status["violations"],
                       "status_if_suggested_exclusions_applied": None if after is None else {
                           "status": after["status"], "code": after["code"], "violations": after["violations"]}},
            "package_profile": _summary(record),
            "notes": ["Nothing was removed or changed by this profile.",
                      "The target, the positive class and every exclusion must be confirmed by the caller."],
        }
