"""Mixed-type table parsing and profiling with limits enforced before heavy work.

This module profiles tables; it does not train, encode or modify them. It uses
only the standard library. pandas is optional: a DataFrame may be passed to
``profile_table`` (it is read through its public ``columns`` and ``to_dict``
interface) and ``table_to_dataframe`` imports pandas lazily, but nothing here
requires it. The numeric-only parser in ``datasets.py`` is unchanged.

Principles:

* The target, group, time and identifier columns are metadata. They are listed as
  excluded and are never predictors.
* No other column and no row is dropped. Constant, empty, duplicate and
  high-cardinality columns are reported; they are not silently removed.
* Byte, row, column, cardinality and expanded-width limits raise typed errors
  before the work they protect (decoding, row materialization, one-hot growth).
* Malformed rows are rejected with an error, never skipped.

The default limits are conservative local defaults chosen for this package. They
are not verified limits of any hosted service.
"""
from __future__ import annotations

import codecs
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import math
from numbers import Real
from collections.abc import Mapping

from .labels import _is_missing

TABLE_TOO_LARGE_BYTES = "TABLE_TOO_LARGE_BYTES"
TABLE_TOO_MANY_ROWS = "TABLE_TOO_MANY_ROWS"
TABLE_TOO_MANY_COLUMNS = "TABLE_TOO_MANY_COLUMNS"
TABLE_CARDINALITY_EXCEEDED = "TABLE_CARDINALITY_EXCEEDED"
TABLE_EXPANSION_EXCEEDED = "TABLE_EXPANSION_EXCEEDED"
TABLE_MALFORMED_ROW = "TABLE_MALFORMED_ROW"
TABLE_ENCODING_UNDECODABLE = "TABLE_ENCODING_UNDECODABLE"
TABLE_ENCODING_UNKNOWN = "TABLE_ENCODING_UNKNOWN"
TABLE_EMPTY = "TABLE_EMPTY"
TABLE_NO_ROWS = "TABLE_NO_ROWS"
TABLE_HEADER_INVALID = "TABLE_HEADER_INVALID"
TABLE_COLUMN_MISSING = "TABLE_COLUMN_MISSING"
TABLE_COLUMN_ROLE_CONFLICT = "TABLE_COLUMN_ROLE_CONFLICT"
TABLE_NO_PREDICTORS = "TABLE_NO_PREDICTORS"
TABLE_INPUT_INVALID = "TABLE_INPUT_INVALID"
TABLE_LIMITS_INVALID = "TABLE_LIMITS_INVALID"
REASON_CODES = (TABLE_TOO_LARGE_BYTES, TABLE_TOO_MANY_ROWS, TABLE_TOO_MANY_COLUMNS, TABLE_CARDINALITY_EXCEEDED,
                TABLE_EXPANSION_EXCEEDED, TABLE_MALFORMED_ROW, TABLE_ENCODING_UNDECODABLE, TABLE_ENCODING_UNKNOWN,
                TABLE_EMPTY, TABLE_NO_ROWS, TABLE_HEADER_INVALID, TABLE_COLUMN_MISSING, TABLE_COLUMN_ROLE_CONFLICT,
                TABLE_NO_PREDICTORS, TABLE_INPUT_INVALID, TABLE_LIMITS_INVALID)

#: Same tokens the numeric parser treats as a missing feature value (after stripping).
MISSING_TOKENS = ("", "?", "NA", "NaN")
DELIMITER_CANDIDATES = (",", ";", "\t", "|")
DELIMITER_SCAN_LINES = 50
COLUMN_KINDS = ("numeric", "categorical", "constant", "high_cardinality", "empty")


class TabularError(ValueError):
    """Typed table failure. ``code`` is a stable reason code; ``details`` is JSON-safe."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details

    def to_record(self) -> dict:
        return {"reason_code": self.code, "message": str(self), "details": self.details}


@dataclass(frozen=True)
class TabularLimits:
    """Upper bounds checked before the work they protect.

    These are conservative local defaults for this package, not verified limits of
    any hosted deployment. ``max_columns`` counts every column, metadata included.
    ``max_expanded_features`` bounds the estimated predictor width after one-hot
    encoding (numeric columns count 1, a categorical column counts its distinct
    non-missing values).
    """

    max_bytes: int = 5_000_000
    max_rows: int = 20_000
    max_columns: int = 205
    max_categorical_cardinality: int = 50
    max_expanded_features: int = 400

    def __post_init__(self):
        for name in ("max_bytes", "max_rows", "max_columns", "max_categorical_cardinality", "max_expanded_features"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TabularError(TABLE_LIMITS_INVALID, f"{name} must be an integer >= 1.", field=name)


@dataclass(frozen=True)
class ParsedTable:
    """Columns as lists of raw cell values plus the parsing record."""

    columns: dict
    row_count: int
    parsing: dict


def _check_columns_and_rows(row_count: int, column_count: int, limits: TabularLimits) -> None:
    if column_count > limits.max_columns:
        raise TabularError(TABLE_TOO_MANY_COLUMNS, f"The table has {column_count} columns; the limit is {limits.max_columns}.",
                           columns=column_count, limit=limits.max_columns)
    if row_count > limits.max_rows:
        raise TabularError(TABLE_TOO_MANY_ROWS, f"The table has {row_count} rows; the limit is {limits.max_rows}.",
                           rows=row_count, limit=limits.max_rows)


def _decide_encoding(raw: bytes, encoding: str | None) -> tuple[str, dict]:
    if encoding is not None:
        try:
            codec = codecs.lookup(encoding)
        except LookupError as exc:
            raise TabularError(TABLE_ENCODING_UNKNOWN, f"Unknown encoding: {encoding!r}.", encoding=str(encoding)) from exc
        try:
            text = raw.decode(codec.name)
        except UnicodeDecodeError as exc:
            raise TabularError(TABLE_ENCODING_UNDECODABLE, f"The bytes are not valid {codec.name}.",
                               encoding=codec.name, byte_offset=exc.start) from exc
        bom = text.startswith("﻿")
        return text.lstrip("﻿") if bom else text, {"encoding": codec.name, "method": "user_specified", "bom_removed": bom}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TabularError(TABLE_ENCODING_UNDECODABLE,
                           "The bytes are not valid UTF-8. No fallback encoding is guessed; pass the correct encoding explicitly.",
                           encoding="utf-8", byte_offset=exc.start) from exc
    bom = text.startswith("﻿")
    return text.lstrip("﻿") if bom else text, {"encoding": "utf-8", "method": "strict_utf8", "bom_removed": bom}


def _decide_delimiter(text: str, delimiter: str | None) -> dict:
    if delimiter is not None:
        if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in "\r\n\"":
            raise TabularError(TABLE_INPUT_INVALID, "delimiter must be one character other than a quote or newline.")
        return {"delimiter": delimiter, "method": "user_specified", "ambiguous": False, "candidates": {}}
    lines = [line for line in text.splitlines()[:DELIMITER_SCAN_LINES] if line.strip()]
    if not lines:
        raise TabularError(TABLE_EMPTY, "The table is empty.")
    scores = {}
    for candidate in DELIMITER_CANDIDATES:
        try:
            widths = [len(row) for row in csv.reader(io.StringIO("\n".join(lines)), delimiter=candidate, strict=True) if row]
        except csv.Error:
            continue
        # Credit a candidate only if it splits the header and every scanned line to the same width.
        if widths and widths[0] > 1 and len(set(widths)) == 1:
            scores[candidate] = widths[0]
    candidates = {("\\t" if key == "\t" else key): value for key, value in scores.items()}
    if not scores:
        return {"delimiter": ",", "method": "default_no_consistent_delimiter", "ambiguous": False, "candidates": {}}
    best = max(scores.values())
    tied = [key for key in DELIMITER_CANDIDATES if scores.get(key) == best]
    # Ambiguous when more than one candidate splits every scanned line consistently; the widest split wins,
    # ties go to the fixed candidate order. The decision is recorded either way.
    return {"delimiter": tied[0], "method": "consistent_field_count_scan", "ambiguous": len(scores) > 1,
            "candidates": candidates}


def parse_table_csv(raw_bytes: bytes, limits: TabularLimits = TabularLimits(), *, delimiter: str | None = None,
                    encoding: str | None = None, source_name: str | None = None) -> ParsedTable:
    """Parse delimited text into raw string columns, rejecting malformed input.

    The byte limit is checked before decoding; the column limit before any row is
    read; the row limit as soon as it is exceeded. A row whose field count differs
    from the header, a blank line before the end of the data, an undecodable byte
    sequence or an unterminated quote raises ``TabularError``; nothing is skipped.
    Blank lines at the very end of the file are counted and recorded.
    """
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise TabularError(TABLE_INPUT_INVALID, "raw_bytes must be bytes.")
    if not isinstance(limits, TabularLimits):
        raise TabularError(TABLE_INPUT_INVALID, "limits must be a TabularLimits instance.")
    raw = bytes(raw_bytes)
    if len(raw) > limits.max_bytes:
        raise TabularError(TABLE_TOO_LARGE_BYTES, f"The input has {len(raw)} bytes; the limit is {limits.max_bytes}.",
                           bytes=len(raw), limit=limits.max_bytes)
    if not raw.strip():
        raise TabularError(TABLE_EMPTY, "The table is empty.")
    text, encoding_record = _decide_encoding(raw, encoding)
    delimiter_record = _decide_delimiter(text, delimiter)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter_record["delimiter"], strict=True)
    try:
        try:
            header = next(reader)
        except StopIteration as exc:
            raise TabularError(TABLE_EMPTY, "The table is empty.") from exc
        if len(header) > limits.max_columns:
            _check_columns_and_rows(0, len(header), limits)
        if any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise TabularError(TABLE_HEADER_INVALID, "Column names must be nonblank and unique.",
                               columns=len(header), line=reader.line_num)
        columns = {name: [] for name in header}
        series = [columns[name] for name in header]
        rows, pending_blank = 0, 0
        for row in reader:
            if not row:
                pending_blank += 1
                continue
            if pending_blank:
                raise TabularError(TABLE_MALFORMED_ROW, f"A blank line occurs before the end of the data (line {reader.line_num - 1}).",
                                   line=reader.line_num - 1, reason="blank_line_inside_data")
            if len(row) != len(header):
                raise TabularError(TABLE_MALFORMED_ROW,
                                   f"Line {reader.line_num} has {len(row)} fields; the header has {len(header)}. "
                                   "Malformed rows are rejected, not dropped.", line=reader.line_num,
                                   fields=len(row), expected=len(header), reason="field_count_mismatch")
            rows += 1
            _check_columns_and_rows(rows, len(header), limits)
            for column, value in zip(series, row):
                column.append(value)
    except csv.Error as exc:
        raise TabularError(TABLE_MALFORMED_ROW, f"The delimited text is malformed near line {reader.line_num}: {exc}",
                           line=reader.line_num, reason="csv_syntax") from exc
    if rows == 0:
        raise TabularError(TABLE_NO_ROWS, "The table has a header but no data rows.")
    parsing = {
        "parser": "parse_table_csv", "source_name": source_name, "input_sha256": sha256(raw).hexdigest(),
        "input_bytes": len(raw), "encoding": encoding_record, "delimiter": delimiter_record,
        "header": list(header), "rows_read": rows, "rows_accepted": rows, "rows_rejected": 0,
        "trailing_blank_lines": pending_blank, "malformed_row_policy": "reject_with_error",
        "row_exclusions": [], "missing_tokens": [token for token in MISSING_TOKENS if token != ""] + ["(blank)"],
    }
    return ParsedTable(columns=columns, row_count=rows, parsing=parsing)


def _as_columns(table, limits: TabularLimits) -> tuple[dict, int, dict | None]:
    """Normalize supported inputs to ``({name: [values]}, rows, parsing_or_None)``, enforcing limits first."""
    if isinstance(table, ParsedTable):
        _check_columns_and_rows(table.row_count, len(table.columns), limits)
        return table.columns, table.row_count, table.parsing
    if hasattr(table, "columns") and hasattr(table, "to_dict"):  # DataFrame-like; no pandas import needed
        names = [str(name) for name in table.columns]
        _check_columns_and_rows(len(table), len(names), limits)
        if len(set(names)) != len(names):
            raise TabularError(TABLE_HEADER_INVALID, "Column names must be unique.")
        data = table.to_dict("list")
        return {str(name): list(values) for name, values in data.items()}, len(table), None
    if isinstance(table, Mapping):
        sizes = {len(values) for values in table.values()}
        if len(sizes) > 1:
            raise TabularError(TABLE_MALFORMED_ROW, "Columns have different lengths.", reason="ragged_columns")
        rows = sizes.pop() if sizes else 0
        _check_columns_and_rows(rows, len(table), limits)
        return {str(name): list(values) for name, values in table.items()}, rows, None
    if isinstance(table, (list, tuple)):
        _check_columns_and_rows(len(table), 0, limits)
        if not table:
            raise TabularError(TABLE_NO_ROWS, "The table has no rows.")
        first = table[0]
        if not isinstance(first, Mapping):
            raise TabularError(TABLE_MALFORMED_ROW, "Row 0 is not a mapping.", row=0, reason="row_not_mapping")
        header = list(first)
        _check_columns_and_rows(len(table), len(header), limits)
        keys = set(header)
        columns = {str(name): [] for name in header}
        for index, row in enumerate(table):
            if not isinstance(row, Mapping) or set(row) != keys:
                raise TabularError(TABLE_MALFORMED_ROW,
                                   f"Row {index} does not have exactly the columns of row 0. Rows are rejected, not repaired.",
                                   row=index, reason="row_keys_differ")
            for name in header:
                columns[str(name)].append(row[name])
        return columns, len(table), None
    raise TabularError(TABLE_INPUT_INVALID, "table must be a ParsedTable, DataFrame, dict of columns or list of dict rows.")


def _number(value):
    """Return a float for a numeric cell, None if it is not numeric. Booleans are categories."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _is_missing_cell(value) -> bool:
    if _is_missing(value):
        return True
    return isinstance(value, str) and value.strip() in MISSING_TOKENS


_MISSING_CELL = "\x00missing"


def _profile_column(values: list, cap: int, row_hashes: list | None = None) -> dict:
    """One pass over a column. With ``row_hashes`` it also updates per-row pattern hashes in place
    and returns a column fingerprint, so duplicate patterns need no second pass or copied cells."""
    missing = 0
    numeric_only, non_finite = True, 0
    seen: dict[str, None] = {}
    capped = False
    fingerprint = 0
    for index, value in enumerate(values):
        if _is_missing_cell(value):
            missing += 1
            text = _MISSING_CELL
        else:
            number = _number(value)
            if number is None:
                numeric_only = False
                text = value if isinstance(value, str) else str(value)
            else:
                if not math.isfinite(number):
                    non_finite += 1
                text = repr(0.0 if number == 0 else number)
            if not capped:
                seen[text] = None
                if len(seen) >= cap:
                    capped = True
        if row_hashes is not None:
            row_hashes[index] = hash((row_hashes[index], text))
            fingerprint = hash((fingerprint, text))
    present = len(values) - missing
    value_type = None if present == 0 else ("numeric" if numeric_only else "categorical")
    return {"missing": missing, "present": present, "distinct": len(seen), "distinct_capped": capped,
            "value_type": value_type, "non_finite_values": non_finite, "fingerprint": fingerprint,
            "missing_fraction": (missing / len(values)) if values else 0.0}


def _normalized_cells(values: list) -> list:
    out = []
    for value in values:
        if _is_missing_cell(value):
            out.append(_MISSING_CELL)
            continue
        number = _number(value)
        out.append(repr(0.0 if number == 0 else number) if number is not None else (value if isinstance(value, str) else str(value)))
    return out


def _distinct_cap(limits: TabularLimits) -> int:
    return max(limits.max_categorical_cardinality + 1, 2)


def profile_table(table, *, target: str, group_column: str | None = None, time_column: str | None = None,
                  id_columns=(), limits: TabularLimits = TabularLimits()) -> dict:
    """Profile columns and enforce limits; return a JSON-safe record.

    Raises ``TabularError`` (with ``details['violations']`` listing every limit
    that failed) if a limit is exceeded. The limits on rows and columns are
    checked before any column is scanned, and distinct-value counting stops once a
    column passes the cardinality limit, so rejection does not require full
    materialization or one-hot expansion.
    """
    if not isinstance(limits, TabularLimits):
        raise TabularError(TABLE_INPUT_INVALID, "limits must be a TabularLimits instance.")
    columns, rows, parsing = _as_columns(table, limits)
    names = list(columns)
    if rows < 1:
        raise TabularError(TABLE_NO_ROWS, "The table has no rows.")
    ids = tuple(id_columns)
    roles: dict[str, str] = {}
    for role, selected in (("target", [target]), ("group", [group_column] if group_column is not None else []),
                           ("time", [time_column] if time_column is not None else []), ("id", list(ids))):
        for name in selected:
            if name not in columns:
                raise TabularError(TABLE_COLUMN_MISSING, f"The {role} column {name!r} is not in the table.",
                                   role=role, column=str(name))
            if name in roles:
                raise TabularError(TABLE_COLUMN_ROLE_CONFLICT,
                                   f"Column {name!r} is assigned to both {roles[name]} and {role}; each column has one role.",
                                   column=str(name), roles=[roles[name], role])
            roles[name] = role
    predictors = [name for name in names if name not in roles]
    predictors_set = set(predictors)
    if not predictors:
        raise TabularError(TABLE_NO_PREDICTORS, "No predictor columns remain after excluding the target, group, time and id columns.")

    cap = _distinct_cap(limits)
    row_hashes = [0] * rows
    stats = {name: _profile_column(columns[name], cap, row_hashes if name in predictors_set else None) for name in names}
    violations = []
    column_records, constants, empties, expanded_total = {}, [], [], 0
    numeric_width = categorical_width = 0
    for name in names:
        info = stats[name]
        role = roles.get(name, "predictor")
        record = {"role": role, "missing": info["missing"], "missing_fraction": info["missing_fraction"],
                  "distinct": info["distinct"], "distinct_capped": info["distinct_capped"],
                  "value_type": info["value_type"], "non_finite_values": info["non_finite_values"]}
        if role == "predictor":
            if info["present"] == 0:
                kind, width = "empty", 1
                empties.append(name)
            elif info["distinct"] <= 1:
                kind, width = "constant", 1
                constants.append(name)
            elif info["value_type"] == "numeric":
                kind, width = "numeric", 1
            elif info["distinct"] > limits.max_categorical_cardinality:
                kind, width = "high_cardinality", info["distinct"]
                violations.append({"code": TABLE_CARDINALITY_EXCEEDED, "column": name, "limit": limits.max_categorical_cardinality,
                                   "distinct_at_least": info["distinct"] if info["distinct_capped"] else None,
                                   "distinct": None if info["distinct_capped"] else info["distinct"]})
            else:
                kind, width = "categorical", info["distinct"]
            record.update(kind=kind, expanded_width=width)
            if kind == "categorical":
                categorical_width += width
            elif kind != "high_cardinality":
                numeric_width += width
            expanded_total += width
        else:
            record.update(kind=None, expanded_width=0)
        column_records[name] = record
    if expanded_total > limits.max_expanded_features:
        violations.append({"code": TABLE_EXPANSION_EXCEEDED, "limit": limits.max_expanded_features,
                           "estimated_width": expanded_total})
    if violations:
        first = violations[0]
        message = ("Predictor column(s) exceed the categorical cardinality limit: "
                   + ", ".join(repr(v["column"]) for v in violations if v["code"] == TABLE_CARDINALITY_EXCEEDED)
                   if first["code"] == TABLE_CARDINALITY_EXCEEDED else
                   f"The estimated one-hot width {expanded_total} exceeds the limit {limits.max_expanded_features}.")
        raise TabularError(first["code"], message + " Exclude the column(s) explicitly if they are identifiers; none were dropped.",
                           violations=violations, limits=asdict(limits))

    excluded = [{"column": name, "role": roles[name], "missing": stats[name]["missing"], "distinct": stats[name]["distinct"],
                 "distinct_capped": stats[name]["distinct_capped"],
                 "reason": "metadata column; never used as a predictor"} for name in names if name in roles]
    target_present = rows - stats[target]["missing"]
    target_counts = None
    if not stats[target]["distinct_capped"]:
        target_counts = dict(sorted(Counter(columns[target][i] if isinstance(columns[target][i], str) else str(columns[target][i])
                                            for i in range(rows) if not _is_missing(columns[target][i])).items()))
    # Duplicate predictor patterns and identical predictor columns: hash-based diagnostics, not leakage claims.
    repeated = [count for count in Counter(row_hashes).values() if count > 1]
    by_fingerprint: dict[int, list[str]] = {}
    for name in predictors:
        by_fingerprint.setdefault(stats[name]["fingerprint"], []).append(name)
    duplicate_columns = []
    for group in by_fingerprint.values():
        if len(group) > 1:  # confirm candidates against the cell values
            cells = {name: _normalized_cells(columns[name]) for name in group}
            confirmed: dict[tuple, list[str]] = {}
            for name in group:
                confirmed.setdefault(tuple(cells[name]), []).append(name)
            duplicate_columns.extend(match for match in confirmed.values() if len(match) > 1)
    return {
        "schema": "table-profile-1.0",
        "rows": rows, "columns_total": len(names), "predictor_count": len(predictors),
        "predictors": predictors, "excluded_columns": excluded,
        "columns": column_records,
        "target_summary": {"column": target, "missing": stats[target]["missing"], "present": target_present,
                           "distinct": stats[target]["distinct"], "distinct_capped": stats[target]["distinct_capped"],
                           "value_counts": target_counts},
        "constant_columns": constants, "empty_columns": empties,
        "duplicate_columns": duplicate_columns,
        "duplicate_feature_groups": {"groups": len(repeated), "rows_in_groups": sum(repeated),
                                     "basis": "exact predictor values (hash of normalized cells), missing values treated as equal",
                                     "interpretation": "Repeated patterns are a diagnostic, not automatic proof of leakage."},
        "expanded_feature_estimate": {"numeric_like": numeric_width, "categorical_one_hot": categorical_width,
                                      "total": expanded_total,
                                      "basis": "numeric, constant and empty columns count 1; categorical columns count their distinct "
                                               "non-missing values (an upper bound for any one training partition)"},
        "limits": asdict(limits), "limits_checked": True, "limit_violations": [],
        "parsing": parsing,
        "notes": ["No predictor column or row was removed.",
                  "Excluded columns are metadata and are never predictors."],
    }


def excluded_metadata_records(target: str, group_column: str | None = None, time_column: str | None = None,
                              id_columns=()) -> list[dict]:
    """Excluded-column records for reports when no table profile is available."""
    records = [{"column": target, "role": "target"}]
    if group_column is not None:
        records.append({"column": group_column, "role": "group"})
    if time_column is not None:
        records.append({"column": time_column, "role": "time"})
    records.extend({"column": name, "role": "id"} for name in id_columns)
    names = [record["column"] for record in records]
    if len(set(names)) != len(names):
        raise TabularError(TABLE_COLUMN_ROLE_CONFLICT, "A column cannot have more than one metadata role.")
    for record in records:
        record["reason"] = "metadata column; never used as a predictor"
    return records


def table_to_dataframe(table: ParsedTable):
    """Return a pandas DataFrame of string cells. Needs the optional 'app' extra (pandas)."""
    try:
        import pandas
    except ImportError as exc:
        raise ImportError("table_to_dataframe needs pandas. Install the package with [app].") from exc
    return pandas.DataFrame(table.columns)
