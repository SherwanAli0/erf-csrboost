"""Request and response models for the HTTP service (``backend/api.py``).

Request models refuse unknown fields. Labels may be text or numbers (the service layer accepts both), but
never booleans, null or containers. These models only check the shape of a request; the package and the job
coordinator check the meaning (an unknown model id, a missing column) and report it as a failed job.

Errors always have one shape, ``{"error": {"code", "message", "details", "request_id"}}``. Messages and details
come from the coordinator or from this module's own fixed texts, never from an exception's text, so they hold no
traceback, no local path and no capability.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

# JSON numbers that are finite floats; ``allow_inf_nan=False`` also rejects the NaN and Infinity constants.
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Label = Union[StrictStr, StrictInt, FiniteFloat]
ColumnName = Annotated[StrictStr, Field(min_length=1, max_length=256)]
ModelId = Annotated[StrictStr, Field(min_length=1, max_length=64)]


class StrictModel(BaseModel):
    """Base for request bodies: an unknown field is a 422, nothing is coerced."""

    model_config = ConfigDict(extra="forbid", str_max_length=4096)


class CostsModel(StrictModel):
    false_positive_cost: FiniteFloat | StrictInt
    false_negative_cost: FiniteFloat | StrictInt
    review_cost: Optional[Union[FiniteFloat, StrictInt]] = None


class JobRequest(StrictModel):
    """Body of ``POST /api/v1/jobs``. The dataset capability travels in the ``Authorization`` header."""

    dataset_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    target: ColumnName
    positive_label: Label
    negative_label: Optional[Label] = None
    one_vs_rest: StrictBool = False
    models: Optional[Annotated[list[ModelId], Field(min_length=1, max_length=50)]] = None
    split_mode: Optional[Annotated[StrictStr, Field(max_length=64)]] = None
    seed: Optional[StrictInt] = None
    calibration: Optional[Annotated[StrictStr, Field(max_length=64)]] = None
    id_columns: Optional[Annotated[list[ColumnName], Field(max_length=1000)]] = None
    exclude_columns: Optional[Annotated[list[ColumnName], Field(max_length=1000)]] = None
    group_column: Optional[ColumnName] = None
    time_column: Optional[ColumnName] = None
    missing_policy: Optional[Annotated[StrictStr, Field(max_length=64)]] = None
    review_fractions: Optional[Annotated[list[Union[FiniteFloat, StrictInt]], Field(max_length=10)]] = None
    costs: Optional[CostsModel] = None

    def job_parameters(self) -> dict:
        """The fields the coordinator takes (no ``dataset_id``, no unset or null value)."""
        values = self.model_dump(exclude={"dataset_id"}, exclude_none=True)
        if not values.get("one_vs_rest"):
            values.pop("one_vs_rest", None)
        return values


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class DatasetCreated(BaseModel):
    dataset_id: str
    capability: str
    profile: dict
    limits: dict
    expires_at: str


class JobAccepted(BaseModel):
    job_id: str
    capability: str
    state: str
    poll_url: str


class Deleted(BaseModel):
    deleted: bool = True


def validation_details(errors: list) -> dict:
    """Field locations and error kinds from a pydantic error list, without the offending input values."""
    items: list[dict[str, Any]] = []
    unknown: list[str] = []
    for item in errors[:20]:
        location = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
        kind = str(item.get("type", "invalid"))
        if kind == "extra_forbidden":
            unknown.append(location)
        items.append({"field": location[:200], "type": kind, "message": _short(str(item.get("msg", "")))})
    details: dict[str, Any] = {"errors": items}
    if unknown:
        details["unknown_fields"] = unknown[:10]
    return details


def _short(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    return first if len(first) <= 120 else first[:117] + "..."
