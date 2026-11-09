"""Typed errors for the application service layer.

Every failure that leaves the service is a ``ServiceError`` with a stable ``code``, a short
message and JSON-safe details. Messages never contain tracebacks or local file system paths.
Errors raised by the evaluation package keep the package's own stable code and are marked
``layer='package'``; refusals from the legacy input readers are marked ``layer='legacy_input'``.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import re

APP_INPUT_INVALID = "APP_INPUT_INVALID"
APP_TARGET_REQUIRED = "APP_TARGET_REQUIRED"
APP_FORMAT_UNSUPPORTED = "APP_FORMAT_UNSUPPORTED"
APP_INPUT_UNREADABLE = "APP_INPUT_UNREADABLE"
APP_ENGINE_UNAVAILABLE = "APP_ENGINE_UNAVAILABLE"
APP_PACKAGE_UNAVAILABLE = "APP_PACKAGE_UNAVAILABLE"
APP_EVALUATION_INVALID = "APP_EVALUATION_INVALID"
APP_REPORT_NOT_JSON_SAFE = "APP_REPORT_NOT_JSON_SAFE"
APP_INTERNAL_ERROR = "APP_INTERNAL_ERROR"
APP_ERROR_CODES = (APP_INPUT_INVALID, APP_TARGET_REQUIRED, APP_FORMAT_UNSUPPORTED, APP_INPUT_UNREADABLE,
                   APP_ENGINE_UNAVAILABLE, APP_PACKAGE_UNAVAILABLE, APP_EVALUATION_INVALID,
                   APP_REPORT_NOT_JSON_SAFE, APP_INTERNAL_ERROR)

_PATH = re.compile(r"(?:[A-Za-z]:[\\/][^\s'\"<>|]*)|(?:\\\\[^\s'\"<>|]+)|(?:(?<![\w.])/(?:[\w.\-~]+/)+[\w.\-~]*)")
MAX_MESSAGE_CHARACTERS = 300


def safe_text(value, limit: int = MAX_MESSAGE_CHARACTERS) -> str:
    """First line of ``value`` with file system paths replaced and the length bounded."""
    text = str(value).strip().splitlines()[0] if str(value).strip() else ""
    text = _PATH.sub("<path>", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_safe(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False, default=str))
    except (TypeError, ValueError):
        return {}


class ServiceError(ValueError):
    """Stable, typed service failure. ``code`` identifies the reason; ``details`` is JSON-safe."""

    def __init__(self, code: str, message: str, *, layer: str = "service", details: dict | None = None):
        super().__init__(safe_text(message))
        self.code = code
        self.layer = layer
        self.details = _json_safe(details or {})

    def to_record(self) -> dict:
        return {"error": {"code": self.code, "message": str(self), "layer": self.layer, "details": self.details}}


def from_package_error(exc) -> ServiceError:
    """Keep the package's stable code; sanitize the message."""
    return ServiceError(exc.code, str(exc), layer="package", details=getattr(exc, "details", None))


@contextmanager
def translate_errors():
    """Convert any exception into a ``ServiceError`` without leaking a traceback or a path.

    ``ServiceError`` passes through. The package's typed errors (``TabularError``, ``LabelError``,
    ``MixedError``) keep their codes. Anything else becomes ``APP_INTERNAL_ERROR`` with only the
    exception class name.
    """
    try:
        yield
    except ServiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - this is the boundary that classifies everything
        if isinstance(getattr(exc, "code", None), str) and isinstance(exc, ValueError):
            raise from_package_error(exc) from None
        raise ServiceError(APP_INTERNAL_ERROR, "An internal error occurred. No further details are shown "
                           f"({type(exc).__name__}).") from None
