"""Job configuration. Every value is provisional and can be overridden by a ``FYD_*`` environment variable.

The hosted presets are measured later (see the resource limits document); nothing here is a verified limit
of any deployment. ``JobConfig.from_env`` ignores a variable that does not parse as the right type instead of
failing, and clamps nothing silently except the per-model timeout, which the evaluation package restricts to
1..600 seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import os
from pathlib import Path
import tempfile

MB = 1024 * 1024


@dataclass(frozen=True)
class JobConfig:
    max_active_heavy: int = 1
    max_waiting_heavy: int = 1
    max_jobs_per_dataset: int = 1
    max_datasets: int = 4
    upload_limit_bytes: int = 8 * MB
    max_rows: int = 10_000
    max_features: int = 100
    model_timeout_seconds: float = 60.0
    job_timeout_seconds: float = 300.0
    queue_wait_seconds: float = 120.0
    light_timeout_seconds: float = 20.0
    light_max_waiting: int = 2
    dataset_idle_seconds: float = 30 * 60.0
    finished_job_seconds: float = 30 * 60.0
    tombstone_seconds: float = 24 * 3600.0
    max_stored_bytes: int = 64 * MB
    max_models: int = 5
    max_result_bytes: int = 8 * MB
    max_legacy_result_bytes: int = 32 * MB
    child_memory_mb: int = 700
    sweep_interval_seconds: float = 5.0
    kill_grace_seconds: float = 10.0
    temp_root: str | None = None
    allow_test_commands: bool = False

    def __post_init__(self):
        for name in ("max_active_heavy", "max_waiting_heavy", "max_jobs_per_dataset", "max_datasets",
                     "upload_limit_bytes", "max_rows", "max_features", "light_max_waiting", "max_stored_bytes",
                     "max_models", "max_result_bytes", "max_legacy_result_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name in
                                                                              ("max_waiting_heavy",
                                                                               "light_max_waiting") else 1):
                raise ValueError(f"{name} must be a non-negative integer (at least 1 for most settings).")
        if isinstance(self.child_memory_mb, bool) or not isinstance(self.child_memory_mb, int) or self.child_memory_mb < 0:
            raise ValueError("child_memory_mb must be a non-negative integer (0 turns the limit off).")
        for name in ("model_timeout_seconds", "job_timeout_seconds", "queue_wait_seconds", "light_timeout_seconds",
                     "dataset_idle_seconds", "finished_job_seconds", "tombstone_seconds", "sweep_interval_seconds",
                     "kill_grace_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f"{name} must be a positive number.")
        if not 1 <= self.model_timeout_seconds <= 600:
            raise ValueError("model_timeout_seconds must be between 1 and 600 (the package range).")

    @property
    def child_memory_bytes(self) -> int:
        """Memory ceiling for one child process tree in bytes; 0 means no ceiling."""
        return self.child_memory_mb * MB

    @property
    def light_wait_seconds(self) -> float:
        """Longest a caller waits for the light lane: everyone ahead of it, then its own run, plus a margin."""
        return self.light_timeout_seconds * (self.light_max_waiting + 1) + 5.0

    @property
    def heavy_capacity(self) -> int:
        return self.max_active_heavy + self.max_waiting_heavy

    def resolved_temp_root(self) -> Path:
        return Path(self.temp_root) if self.temp_root else Path(tempfile.gettempdir()) / "fyd-jobs"

    def limits_summary(self) -> dict:
        """Numbers a client may see (no paths)."""
        return {"upload_limit_bytes": self.upload_limit_bytes, "max_rows": self.max_rows,
                "max_features": self.max_features, "max_models_per_job": self.max_models,
                "per_model_timeout_seconds": self.model_timeout_seconds,
                "job_timeout_seconds": self.job_timeout_seconds, "queue_wait_seconds": self.queue_wait_seconds,
                "dataset_idle_seconds": self.dataset_idle_seconds,
                "finished_job_seconds": self.finished_job_seconds}

    def tabular_limits_kwargs(self) -> dict:
        """Arguments for ``rare_event_eval.tabular.TabularLimits``; other fields keep the package defaults."""
        return {"max_bytes": self.upload_limit_bytes, "max_rows": self.max_rows,
                "max_expanded_features": self.max_features}

    @classmethod
    def from_env(cls, environ=None, **overrides) -> "JobConfig":
        """Read ``FYD_<FIELD>`` variables (for example ``FYD_MAX_DATASETS``); explicit overrides win."""
        environ = os.environ if environ is None else environ
        values = {}
        for item in fields(cls):
            raw = environ.get("FYD_" + item.name.upper())
            if raw is None or raw == "":
                continue
            try:
                if item.name == "allow_test_commands":
                    values[item.name] = raw.strip() == "1"
                elif item.name == "temp_root":
                    values[item.name] = raw
                elif item.type in ("int", int):
                    values[item.name] = int(raw)
                else:
                    values[item.name] = float(raw)
            except ValueError:
                continue
        values.update(overrides)
        return replace(cls(), **values)
