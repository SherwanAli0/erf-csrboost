"""Input preparation for the application service.

Uploads (bytes) go through the legacy readers in ``backend/engine.py`` (``any_to_csv`` for ZIP, GZIP,
Excel, Stata, JSON and text encodings, then ``read_csv`` for delimiters, headers, number formats and
missing-value spellings). That keeps every table-shaped input the live app accepts working. The result
is wrapped as a package ``ParsedTable`` whose parsing record lists the conversion notes and the legacy
read warnings, so nothing the reader changed is hidden. Table objects (``ParsedTable``, DataFrame,
dict of columns) skip the legacy readers and go to the package unchanged.

The legacy engine is imported lazily and only for uploads and for the target suggestion. It needs
pandas (the ``app`` extra). Importing this module does not import the engine, pandas or the live server.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import sys

from .errors import (APP_ENGINE_UNAVAILABLE, APP_FORMAT_UNSUPPORTED, APP_INPUT_INVALID, APP_INPUT_UNREADABLE,
                     ServiceError, safe_text)
from .evaluation import _package

BACKEND_DIRECTORY = Path(__file__).resolve().parent.parent
TABLE_TOO_LARGE_BYTES = "TABLE_TOO_LARGE_BYTES"
_SKIPPED = re.compile(r"Skipped (\d+) malformed row")


@dataclass(frozen=True)
class PreparedInput:
    """Columns as lists of raw cells plus the parsing record; ``frame`` is the legacy DataFrame, if any."""

    table: object          # a package ParsedTable, or the caller's table object
    columns: dict
    rows: int
    parsing: dict | None
    frame: object | None   # pandas DataFrame from the legacy reader, or the caller's DataFrame


def legacy_engine():
    """Import ``backend/engine.py`` on first use. Raises ``APP_ENGINE_UNAVAILABLE`` if pandas is missing."""
    try:
        if str(BACKEND_DIRECTORY) not in sys.path:
            sys.path.insert(0, str(BACKEND_DIRECTORY))
        import engine
    except ImportError as exc:
        raise ServiceError(APP_ENGINE_UNAVAILABLE,
                           "The legacy input readers need pandas and imbalanced-learn (install the 'app' extra "
                           "and the repository requirements).") from exc
    return engine


def _legacy_refusal(exc: ValueError) -> ServiceError:
    message = safe_text(exc)
    code = APP_FORMAT_UNSUPPORTED if "not supported" in message.lower() else APP_INPUT_UNREADABLE
    return ServiceError(code, message, layer="legacy_input")


def normalize_upload(raw: bytes, filename: str, limits, import_selection=None) -> PreparedInput:
    """Convert an upload to a ``ParsedTable`` through the legacy readers, with byte limits before and after."""
    _package()
    from rare_event_eval.tabular import ParsedTable
    if not isinstance(raw, (bytes, bytearray)):
        raise ServiceError(APP_INPUT_INVALID, "An upload must be bytes.")
    raw = bytes(raw)
    if len(raw) > limits.max_bytes:
        raise ServiceError(TABLE_TOO_LARGE_BYTES, f"The input has {len(raw)} bytes; the limit is {limits.max_bytes}.",
                           layer="package", details={"bytes": len(raw), "limit": limits.max_bytes, "stage": "upload"})
    if not raw.strip():
        raise ServiceError(APP_INPUT_UNREADABLE, "The file is empty.", layer="legacy_input")
    engine = legacy_engine()
    name = filename if isinstance(filename, str) else ""
    try:
        from .imports import convert
        csv_text, notes = convert(raw, name, limits, engine, import_selection)
        converted_bytes = len(csv_text.encode("utf-8", errors="replace"))
        if converted_bytes > limits.max_bytes:
            raise ServiceError(TABLE_TOO_LARGE_BYTES,
                               f"The converted table has {converted_bytes} bytes; the limit is {limits.max_bytes}.",
                               layer="package", details={"bytes": converted_bytes, "limit": limits.max_bytes,
                                                         "stage": "converted"})
        frame = engine.read_csv(csv_text)
    except ServiceError:
        raise
    except ValueError as exc:
        raise _legacy_refusal(exc) from None
    except ImportError as exc:
        raise ServiceError(APP_ENGINE_UNAVAILABLE, f"A reader dependency is missing ({safe_text(exc)}).") from None
    except Exception as exc:  # noqa: BLE001 - parser libraries raise many types; none may leak a trace or a path
        raise ServiceError(APP_INPUT_UNREADABLE, f"The file could not be read as a table ({type(exc).__name__}).",
                           layer="legacy_input") from None
    warnings = [str(item) for item in frame.attrs.get("read_warnings", [])]
    skipped = next((int(m.group(1)) for item in warnings for m in [_SKIPPED.search(item)] if m), None)
    columns = {str(column): frame[column].tolist() for column in frame.columns}
    parsing = {
        "parser": "legacy_engine.read_csv", "source_name": None, "input_sha256": sha256(raw).hexdigest(),
        "input_bytes": len(raw), "converted_csv_bytes": converted_bytes, "format_notes": [str(item) for item in notes],
        "read_warnings": warnings, "rows_read": len(frame), "rows_accepted": len(frame),
        "malformed_rows_skipped_by_reader": skipped, "rows_rejected": 0,
        "malformed_row_policy": "legacy reader: tolerant; skipped rows are counted in read_warnings",
        "row_exclusions": [],
        "import_selection": dict(import_selection or {}),
        "missing_tokens": ["(blank)", "?", "N/A", "n/a", "plus the default missing-value spellings of pandas"],
        "note": "Delimiter, header, number-format and missing-value handling were decided by the legacy reader "
                "on the whole table before any split; no label or fitted statistic is used by it.",
    }
    table = ParsedTable(columns=columns, row_count=len(frame), parsing=parsing)
    return PreparedInput(table=table, columns=columns, rows=len(frame), parsing=parsing, frame=frame)


def prepare_input(data, filename: str, limits, import_selection=None) -> PreparedInput:
    """Upload bytes go through the legacy readers; table objects go to the package unchanged."""
    if isinstance(data, (bytes, bytearray)):
        return normalize_upload(data, filename, limits, import_selection)
    from rare_event_eval.tabular import _as_columns
    columns, rows, parsing = _as_columns(data, limits)
    frame = data if hasattr(data, "columns") and hasattr(data, "to_dict") else None
    return PreparedInput(table=data, columns=columns, rows=rows, parsing=parsing, frame=frame)
