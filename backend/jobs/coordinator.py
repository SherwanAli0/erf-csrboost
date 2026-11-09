"""The in-process coordinator: datasets, jobs, queue, lanes, ownership, lifetimes.

One supervisor thread owns every child process. A heavy lane runs ``max_active_heavy`` jobs (default one) with
``max_waiting_heavy`` waiting (default one); a light lane runs one short parsing task at a time with a small
waiting line. Every data-touching task is a child process (``runner.py``) in its own process group or job, so
a cancel or a timeout is a real tree kill and a crash cannot take the web process down.

The public methods are synchronous and thread-safe. Failures are ``JobError(code, http_status, message,
details)`` with text that holds no traceback, no path, no token and no raw row.

After a restart nothing survives: ``start`` deletes the whole private storage root, so jobs and datasets of
the previous process are simply unknown (the API cannot name them).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time

from . import proc, states
from .config import JobConfig
from .store import Store, StoreError, read_json_bounded, remove_tree
from .tokens import check_capability, new_capability, new_id

log = logging.getLogger("fyd.jobs")

RUNNER_PATH = Path(__file__).resolve().parent / "runner.py"
LIGHT_COMMANDS = ("profile", "legacy_profile", "legacy_convert")
HEAVY_COMMANDS = ("train", "legacy_analyze", "legacy_score", "legacy_tools")
TEST_COMMANDS = ("test_sleep", "test_spawn", "test_crash", "test_big_output", "test_env", "test_memory",
                 "test_legacy_big_output")
MEMORY_LIMIT_REASON = {"code": "MEMORY_LIMIT_EXCEEDED", "message":
                       "This request needs more memory than the service allows for one job. Try a smaller file."}
LEGACY_PROFILE_RESERVE = 4 * 1024 * 1024   # a legacy profile result holds statistics only
EXPORT_KINDS = ("report_json", "report_html", "config_json", "repro_bundle")
TRAIN_FIELDS = ("target", "positive_label", "negative_label", "one_vs_rest", "models", "split_mode", "seed",
                "calibration", "id_columns", "exclude_columns", "group_column", "time_column", "missing_policy",
                "review_fractions", "costs")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS")
_SECRET_ENVIRONMENT = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE)
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_PATH = re.compile(r"(?:[A-Za-z]:[\\/][^\s'\"<>|]*)|(?:\\\\[^\s'\"<>|]+)|(?:(?<![\w.])/(?:[\w.\-~]+/)+[\w.\-~]*)")
MAX_PROGRESS_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 64 * 1024
TERMINAL_WITH_DIRECTORY = (states.SUCCEEDED, states.PARTIAL)
INTERNAL_CODES = ("JOB_CRASHED", "JOB_INTERNAL_ERROR", "APP_INTERNAL_ERROR", "JOB_START_FAILED")


def _safe(text, limit: int = 300) -> str:
    text = str(text).strip().splitlines()[0] if str(text).strip() else ""
    text = _PATH.sub("<path>", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class JobError(Exception):
    """A typed refusal or failure. ``message`` and ``details`` are safe to return to a client."""

    def __init__(self, code: str, http_status: int, message: str, details: dict | None = None):
        message = _safe(message)
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


def _not_found() -> JobError:
    return JobError("NOT_FOUND", 404, "Not found.")


def is_legacy_command(command: str) -> bool:
    """The four legacy commands, and the test command that stands in for them, use the legacy result cap."""
    return command.startswith("legacy_") or command == "test_legacy_big_output"


def size_text(size: int) -> str:
    """``32 MB``, ``1.5 MB`` or ``200 KB``: a limit as a person reads it."""
    if size < 1024 * 1024:
        return f"{max(1, size // 1024)} KB"
    text = f"{size / (1024 * 1024):.1f}"
    return (text[:-2] if text.endswith(".0") else text) + " MB"



def _services():
    """The application service layer (``backend/services``), imported on first use."""
    backend = str(Path(__file__).resolve().parent.parent)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    import services
    return services


def _short(identifier) -> str:
    return str(identifier)[:8]


def _iso(wall: float) -> str:
    return datetime.fromtimestamp(wall, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------ child launch

def child_environment(config: JobConfig, temp_dir=None) -> dict:
    """Environment for a child: thread limits, no secret-looking variables, the parent pid for the watchdog.

    ``temp_dir`` becomes the child's temporary directory, so the package's per-model worker directories live
    inside the job directory and are deleted with it.
    """
    env = {key: value for key, value in os.environ.items() if not _SECRET_ENVIRONMENT.search(key)}
    env.update({name: "1" for name in THREAD_VARIABLES})
    env.update({"LOKY_MAX_CPU_COUNT": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
                "FYD_PARENT_PID": str(os.getpid()), "FYD_MAX_RESULT_BYTES": str(config.max_result_bytes),
                "FYD_MAX_LEGACY_RESULT_BYTES": str(config.max_legacy_result_bytes),
                "FYD_CHILD_MEMORY_MB": str(config.child_memory_mb)})
    if temp_dir is not None:
        env.update({"TMP": str(temp_dir), "TEMP": str(temp_dir), "TMPDIR": str(temp_dir)})
    if config.allow_test_commands:
        env["FYD_ALLOW_TEST_COMMANDS"] = "1"
    else:
        env.pop("FYD_ALLOW_TEST_COMMANDS", None)
    return env


def spawn_runner(config: JobConfig, command: str, job_dir: Path, runner_path=None, use_job: bool = True):
    """Start ``runner.py <command> <job_dir>`` as a child (shared by the coordinator and the tests)."""
    job_dir = Path(job_dir)
    temp_dir = job_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)
    argv = [proc.python_executable(), "-B", str(runner_path or RUNNER_PATH), command, str(job_dir)]
    return proc.start_child(argv, cwd=job_dir, env=child_environment(config, temp_dir), use_job=use_job,
                            memory_limit_bytes=config.child_memory_bytes)


# ------------------------------------------------------------------------------------------ records

class _Dataset:
    def __init__(self, dataset_id, digest, directory, size, filename, now):
        self.id, self.digest, self.dir, self.bytes, self.filename = dataset_id, digest, directory, size, filename
        self.last_used = now
        self.created_wall = time.time()


class _Job:
    def __init__(self, job_id, digest, kind, lane, command, directory, dataset_id, now, timeout):
        self.id, self.digest, self.kind, self.lane = job_id, digest, kind, lane
        self.command, self.dir, self.dataset_id = command, directory, dataset_id
        self.state = states.QUEUED
        self.created = self.queued_at = now
        self.started_at = self.finished_at = self.expires_at = None
        self.timeout = timeout
        self.process = None
        self.cancel_requested = False
        self.reason = None               # {"code", "message"}
        self.models_selected = []
        self.models = []                 # [{"model", "status", "reason_code"?}]
        self.progress = None             # last sanitized progress snapshot
        self.params = None               # validated train request (for config export)
        self.payload = None              # light and legacy results, held in memory
        self.elapsed = None
        self.done = threading.Event()    # set when the job reached a terminal state
        self.exit_code = None
        self.reserve = 0                 # result bytes this job may still add to storage (legacy jobs only)

    @property
    def live(self) -> bool:
        return self.state in states.LIVE_STATES


class HeavyHandle:
    """Handle for a legacy heavy job (anonymous: no tokens)."""

    def __init__(self, coordinator: "Coordinator", job: _Job):
        self._coordinator, self._job = coordinator, job

    @property
    def state(self) -> str:
        return self._job.state

    def wait(self, timeout: float | None = None) -> dict:
        """Block until the job ends. Returns the child's result; raises ``JobError`` for any other outcome."""
        if not self._job.done.wait(timeout):
            raise JobError("WAIT_TIMEOUT", 503, "The job is still running.")
        return self._coordinator._legacy_outcome(self._job)

    def cancel(self) -> None:
        """Cancel the job (idempotent). Returns after the process tree is gone."""
        self._coordinator._cancel(self._job)


# ------------------------------------------------------------------------------------------ coordinator

class Coordinator:
    def __init__(self, config: JobConfig, runner_path=None):
        self.config = config
        self.runner_path = Path(runner_path) if runner_path else RUNNER_PATH
        self.store = Store(config.resolved_temp_root())
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._started = False
        self._stopped = False
        self._datasets: dict = {}
        self._jobs: dict = {}
        self._tombstones: dict = {}      # id -> (kind, digest or None, expires)
        self._heavy_queue: deque = deque()
        self._heavy_active: list = []
        self._light_queue: deque = deque()
        self._light_active = None
        self._creating = 0
        self._pending_delete: set = set()
        self._next_sweep = 0.0
        self._clock = time.monotonic
        # Set by the web layer (api.create_app): app version and source release id, written into reports and bundles.
        self.app_context: dict = {"app_version": "unknown", "release_id": "unknown"}

    # ================================================================ lifecycle

    def start(self) -> None:
        """Wipe the private storage root and start the supervisor thread."""
        with self._lock:
            if self._started or self._stopped:
                raise RuntimeError("the coordinator can be started once")
            try:
                self.store.prepare_root()
            except StoreError as exc:
                raise RuntimeError(f"job storage could not be prepared: {exc}") from None
            self._started = True
            self._next_sweep = self._clock() + self.config.sweep_interval_seconds
            self._thread = threading.Thread(target=self._run, name="fyd-job-supervisor", daemon=True)
            self._thread.start()
        log.info("job coordinator started")

    def stop(self, deadline: float = 10) -> None:
        """Cancel everything, kill every process tree, delete the storage root."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
        self._stop.set()
        self._wake.set()
        if thread is not None:
            thread.join(timeout=max(0.1, float(deadline)))
        with self._lock:  # fallback when the supervisor did not finish in time
            for job in list(self._jobs.values()):
                if job.process is not None and job.process.poll() is None:
                    job.process.kill_tree(2.0)
                if not job.done.is_set():
                    self._force_interrupt(job)
            for job in list(self._jobs.values()):
                if job.process is not None:
                    job.process.release()
        if not self.store.destroy_root():
            time.sleep(0.5)
            self.store.destroy_root()
        log.info("job coordinator stopped")

    def stats(self) -> dict:
        """Counts only: no ids, no names, no paths."""
        with self._lock:
            by_state: dict = {}
            for job in self._jobs.values():
                by_state[job.state] = by_state.get(job.state, 0) + 1
            return {"datasets": len(self._datasets), "jobs": len(self._jobs), "jobs_by_state": by_state,
                    "heavy_running": len(self._heavy_active), "heavy_waiting": len(self._heavy_queue),
                    "light_running": 1 if self._light_active is not None else 0,
                    "light_waiting": len(self._light_queue), "tombstones": len(self._tombstones),
                    "stored_bytes": self.store.used_bytes() if self._started and not self._stopped else 0,
                    "started": self._started and not self._stopped}

    def _require_running(self) -> None:
        if not self._started or self._stopped or self._stop.is_set():
            raise JobError("SERVICE_UNAVAILABLE", 503, "The job service is not running.")

    # ================================================================ datasets

    def create_dataset(self, raw: bytes, filename: str) -> dict:
        """Store an upload, profile it in the light lane and return the dataset and its one-time capability."""
        if not isinstance(raw, (bytes, bytearray)):
            raise JobError("UPLOAD_INVALID", 400, "An upload must be bytes.")
        raw = bytes(raw)
        name = self._clean_filename(filename)
        if len(raw) > self.config.upload_limit_bytes:
            raise JobError("UPLOAD_TOO_LARGE", 413, "The upload is larger than the service allows.",
                           {"limit_bytes": self.config.upload_limit_bytes})
        if not raw.strip():
            raise JobError("UPLOAD_EMPTY", 400, "The file is empty.")
        dataset_id = new_id("ds")
        with self._lock:
            self._require_running()
            if len(self._datasets) + self._creating >= self.config.max_datasets:
                raise JobError("DATASET_LIMIT_REACHED", 429, "Too many datasets are stored right now. "
                               "Delete one or try again later.", {"retry_after_seconds": 30})
            if (self.store.used_bytes() + self._live_reserved() + len(raw) + self.config.max_result_bytes
                    > self.config.max_stored_bytes):
                raise JobError("STORAGE_FULL", 503, "The service has no room for another dataset right now.",
                               {"retry_after_seconds": 60})
            directory = self.store.make_dir(self.store.dataset_dir(dataset_id))
            self._creating += 1
        registered = False
        try:
            data_path = directory / "data.bin"
            data_path.write_bytes(raw)
            request = {"data_file": str(data_path), "filename": name, "limits": self._limits_kwargs(),
                       "model_timeout_seconds": None, "job_timeout_seconds": self.config.light_timeout_seconds}
            profile = self._run_light_internal("profile", request, None)
            token, digest = new_capability()
            now = self._clock()
            dataset = _Dataset(dataset_id, digest, directory, len(raw), name, now)
            with self._lock:
                self._require_running()
                self._datasets[dataset_id] = dataset
                registered = True
            log.info("dataset created id=%s", _short(dataset_id))
            return {"dataset_id": dataset_id, "capability": token, "profile": profile,
                    "expires_at": _iso(time.time() + self.config.dataset_idle_seconds),
                    "limits": self.config.limits_summary()}
        finally:
            with self._lock:
                self._creating -= 1
            if not registered:
                self._delete_path(directory)

    def delete_dataset(self, dataset_id: str, capability: str) -> dict:
        with self._lock:
            dataset = self._owned_dataset(dataset_id, capability)
            live = [job for job in self._jobs.values() if job.dataset_id == dataset.id and job.live]
        for job in live:
            self._cancel(job)
        with self._lock:
            self._datasets.pop(dataset.id, None)
        self._delete_path(dataset.dir)
        log.info("dataset deleted id=%s", _short(dataset.id))
        return {"deleted": True}

    def _owned_dataset(self, dataset_id, capability) -> _Dataset:
        """Caller holds the lock. Wrong, foreign, missing and unknown all give the same NOT_FOUND."""
        dataset = self._datasets.get(dataset_id) if isinstance(dataset_id, str) else None
        if dataset is not None:
            if not check_capability(capability, dataset.digest):
                raise _not_found()
            return dataset
        entry = self._tombstones.get(dataset_id) if isinstance(dataset_id, str) else None
        if entry is not None and entry[0] == "dataset" and check_capability(capability, entry[1]):
            raise JobError("DATASET_EXPIRED", 410, "This dataset has expired and its files were deleted.")
        check_capability(capability, None)
        raise _not_found()

    @staticmethod
    def _clean_filename(filename) -> str:
        text = filename if isinstance(filename, str) else ""
        text = re.sub(r"[\x00-\x1f\x7f]", "", text.replace("\\", "/").rsplit("/", 1)[-1])
        return text[:120]

    def _limits_kwargs(self) -> dict:
        return self.config.tabular_limits_kwargs()

    # ================================================================ jobs

    def submit_job(self, dataset_id: str, capability: str, request: dict) -> dict:
        params = self._validate_train_request(request)
        with self._lock:
            self._require_running()
            dataset = self._owned_dataset(dataset_id, capability)
            live = sum(1 for job in self._jobs.values() if job.dataset_id == dataset.id and job.live)
            if live >= self.config.max_jobs_per_dataset:
                raise JobError("DATASET_BUSY", 429, "This dataset already has a job that has not finished.",
                               {"retry_after_seconds": 10})
            self._check_heavy_capacity()
            if (self.store.used_bytes() + self._live_reserved() + self.config.max_result_bytes
                    > self.config.max_stored_bytes):
                raise JobError("STORAGE_FULL", 503, "The service has no room for another job right now.",
                               {"retry_after_seconds": 60})
            job_id = new_id("job")
            directory = self.store.make_dir(self.store.job_dir(job_id))
            token, digest = new_capability()
            now = self._clock()
            job = _Job(job_id, digest, "train", "heavy", "train", directory, dataset.id, now,
                       self.config.job_timeout_seconds)
            job.params = params
            job.models_selected = list(params.get("models") or [])
            body = {"command": "train", "data_file": str(dataset.dir / "data.bin"), "filename": dataset.filename,
                    "limits": self._limits_kwargs(), "params": params,
                    "model_timeout_seconds": self.config.model_timeout_seconds,
                    "job_timeout_seconds": self.config.job_timeout_seconds}
            try:
                (directory / "request.json").write_text(json.dumps(body, allow_nan=False), encoding="utf-8")
            except (OSError, ValueError):
                self._delete_path(directory)
                raise JobError("JOB_REQUEST_INVALID", 422, "The job request could not be stored.") from None
            dataset.last_used = now
            self._jobs[job_id] = job
            self._heavy_queue.append(job)
        self._wake.set()
        log.info("job queued id=%s", _short(job_id))
        return {"job_id": job_id, "capability": token, "state": states.QUEUED}

    def job_status(self, job_id: str, capability: str) -> dict:
        with self._lock:
            job, tombstone = self._owned_job(job_id, capability)
            if tombstone is not None:
                return {"job_id": job_id, "state": states.EXPIRED, "queue_position": None, "progress": None,
                        "elapsed_seconds": None, "limits": self._job_limits(), "warnings": [], "models": [],
                        "reason": None, "expires_at": None}
            return self._status(job)

    def cancel_job(self, job_id: str, capability: str) -> dict:
        with self._lock:
            job, tombstone = self._owned_job(job_id, capability)
        if tombstone is None:
            self._cancel(job)
        return self.job_status(job_id, capability)

    def job_result(self, job_id: str, capability: str):
        """``(state, report)``: the ``training-1.1`` record for succeeded and partial, ``None`` otherwise."""
        with self._lock:
            job, tombstone = self._owned_job(job_id, capability)
            if tombstone is not None:
                return states.EXPIRED, None
            state = job.state
            if state not in states.RESULT_STATES:
                return state, None
            report = self._read_report(job)
            if report is None:
                return states.EXPIRED, None
            return state, self._annotated(report)

    def job_export(self, job_id: str, capability: str, kind: str):
        """``(bytes, content_type, filename)`` for ``report_json``, ``report_html``, ``config_json`` or ``repro_bundle``.

        Every export is built from the stored report and the validated request only: none holds a row of the
        uploaded table. The report carries the provenance label ``fresh app run`` and the app block (see
        ``services.annotate_report``); ``/result`` returns the same report.
        """
        if kind not in EXPORT_KINDS:
            raise JobError("EXPORT_UNKNOWN", 404, "Unknown export.")
        with self._lock:
            job, tombstone = self._owned_job(job_id, capability)
            if tombstone is not None:
                raise JobError("JOB_EXPIRED", 410, "This job has expired and its files were deleted.")
            if job.state not in states.RESULT_STATES:
                raise JobError("RESULT_NOT_AVAILABLE", 409, f"The job is {job.state}; there is no report to export.")
            report = self._read_report(job)
            if report is None:
                raise JobError("JOB_EXPIRED", 410, "This job has expired and its files were deleted.")
            params = dict(job.params or {})
        services = _services()
        report = self._annotated(report, services)
        if kind == "report_json":
            return services.dump_json(report), "application/json", "fyd-report.json"
        if kind == "config_json":
            body = self._config_body(report, params, services)
            return services.dump_json(body), "application/json", "fyd-config.json"
        try:
            html = services.render_report_html(report)
        except Exception:  # noqa: BLE001 - the renderer's own errors must not leak
            raise JobError("EXPORT_FAILED", 500, "The report could not be rendered.") from None
        if kind == "report_html":
            return html.encode("utf-8"), "text/html; charset=utf-8", "fyd-report.html"
        settings = services.effective_settings(params, self.config.tabular_limits_kwargs(),
                                               self.config.model_timeout_seconds)
        try:
            data = services.build_repro_bundle(report, settings, self._config_body(report, params, services), html,
                                               self.app_context)
        except services.ExportError as exc:
            raise JobError(exc.code, 500, str(exc)) from None
        except Exception:  # noqa: BLE001 - never leak the text of an unexpected failure
            raise JobError("EXPORT_FAILED", 500, "The bundle could not be built.") from None
        return data, "application/zip", "fyd-repro-bundle.zip"

    def _annotated(self, report: dict, services=None) -> dict:
        services = services or _services()
        return services.annotate_report(report, services.FRESH_RUN, self.app_context)

    def _config_body(self, report: dict, params: dict, services) -> dict:
        settings = services.effective_settings(params, self.config.tabular_limits_kwargs(),
                                               self.config.model_timeout_seconds)
        return {"schema": "fyd-job-config-1.0", "provenance": report.get("provenance"), "request": params,
                "effective": settings, "limits": self.config.limits_summary(),
                "report_schema_version": report.get("schema_version"), "input_sha256": report.get("input_sha256"),
                "app": report.get("app"),
                "note": "The settings of this run, with every default written out under effective. Run it again "
                        "with the same data to compare. It holds no row of your data."}

    def _owned_job(self, job_id, capability):
        """Caller holds the lock. Returns ``(job, None)`` or ``(None, tombstone)``; otherwise NOT_FOUND."""
        job = self._jobs.get(job_id) if isinstance(job_id, str) else None
        if job is not None and job.digest is not None:
            if not check_capability(capability, job.digest):
                raise _not_found()
            return job, None
        entry = self._tombstones.get(job_id) if isinstance(job_id, str) else None
        if entry is not None and entry[0] == "job" and check_capability(capability, entry[1]):
            return None, entry
        check_capability(capability, None)
        raise _not_found()

    def _job_limits(self) -> dict:
        return {"per_model_timeout_seconds": self.config.model_timeout_seconds,
                "job_timeout_seconds": self.config.job_timeout_seconds,
                "queue_wait_seconds": self.config.queue_wait_seconds}

    def _status(self, job: _Job) -> dict:
        now = self._clock()
        position = None
        if job.state == states.QUEUED and job in self._heavy_queue:
            position = list(self._heavy_queue).index(job) + 1
        if job.elapsed is not None:
            elapsed = job.elapsed
        elif job.started_at is not None:
            elapsed = round(now - job.started_at, 3)
        else:
            elapsed = 0.0
        progress = job.progress
        if job.state == states.RUNNING:
            progress = self._read_progress(job) or progress
        models = list(job.models)
        if job.state == states.RUNNING and progress:
            models = [dict(item) for item in progress.get("models_completed", [])]
        warnings = list(progress.get("warnings", [])) if progress else []
        expires = None
        if job.expires_at is not None:
            expires = _iso(time.time() + (job.expires_at - now))
        return {"job_id": job.id, "state": job.state, "queue_position": position,
                "progress": None if progress is None else {**progress, "elapsed_seconds": elapsed},
                "elapsed_seconds": elapsed, "limits": self._job_limits(), "warnings": warnings,
                "models": models, "models_selected": list(job.models_selected),
                "reason": dict(job.reason) if job.reason else None, "expires_at": expires}

    def _read_progress(self, job: _Job):
        try:
            raw = read_json_bounded(job.dir / "progress.json", MAX_PROGRESS_BYTES)
        except StoreError:
            return None
        return self._sanitize_progress(raw)

    @staticmethod
    def _sanitize_progress(raw):
        """Keep only the documented keys, with bounded sizes and types. The file is untrusted input."""
        if not isinstance(raw, dict) or raw.get("stage") not in states.STAGES:
            return None

        def text(value, limit=64):
            return _safe(value, limit) if isinstance(value, (str, int)) else None

        selected = [text(item) for item in raw.get("models_selected", [])[:50] if text(item)] \
            if isinstance(raw.get("models_selected"), list) else []
        completed = []
        if isinstance(raw.get("models_completed"), list):
            for item in raw["models_completed"][:50]:
                if isinstance(item, dict) and text(item.get("model")) and text(item.get("status")):
                    entry = {"model": text(item["model"]), "status": text(item["status"])}
                    if text(item.get("reason_code")):
                        entry["reason_code"] = text(item["reason_code"])
                    completed.append(entry)
        warnings = [_safe(item, 200) for item in raw.get("warnings", [])[:5] if isinstance(item, str)] \
            if isinstance(raw.get("warnings"), list) else []
        limits = raw.get("limits") if isinstance(raw.get("limits"), dict) else {}
        limits = {key: value for key, value in limits.items()
                  if key in ("model_timeout_seconds", "job_timeout_seconds")
                  and isinstance(value, (int, float)) and not isinstance(value, bool)}
        return {"stage": raw["stage"], "models_selected": selected, "models_completed": completed,
                "current_model": text(raw.get("current_model")), "limits": limits, "warnings": warnings}

    def _read_report(self, job: _Job):
        try:
            report = read_json_bounded(job.dir / "result.json", self.config.max_result_bytes)
        except StoreError:
            return None
        return report if isinstance(report, dict) else None

    # ---------------------------------------------------------------- validation

    def _validate_train_request(self, request) -> dict:
        def bad(message, **details):
            return JobError("REQUEST_INVALID", 422, message, details)

        if not isinstance(request, dict):
            raise bad("The job request must be an object.")
        params = {key: value for key, value in request.items() if key != "dataset_id"}
        unknown = sorted(str(key) for key in params if key not in TRAIN_FIELDS)
        if unknown:
            raise bad("The job request has an unknown field.", unknown_fields=unknown[:10])
        target = params.get("target")
        if not isinstance(target, str) or not target or len(target) > 256:
            raise bad("target must be a column name.")
        label = params.get("positive_label")
        if label is None or isinstance(label, (dict, list)):
            raise bad("positive_label is required.")
        one_vs_rest = params.get("one_vs_rest", False)
        if not isinstance(one_vs_rest, bool):
            raise bad("one_vs_rest must be true or false.")
        if one_vs_rest == (params.get("negative_label") is not None):
            raise bad("Give negative_label for a binary target, or one_vs_rest true with no negative_label.")
        if "models" in params:
            models = params["models"]
            if (not isinstance(models, list) or not models
                    or any(not isinstance(item, str) or not 0 < len(item) <= 64 for item in models)
                    or len(set(models)) != len(models)):
                raise bad("models must be a non-empty list of unique model ids.")
            if len(models) > self.config.max_models:
                raise JobError("TOO_MANY_MODELS", 422, f"A job can run at most {self.config.max_models} models.",
                               {"max_models": self.config.max_models})
        for key in ("id_columns", "exclude_columns"):
            if key in params and (not isinstance(params[key], list) or len(params[key]) > 1000
                                  or any(not isinstance(item, str) for item in params[key])):
                raise bad(f"{key} must be a list of column names.")
        for key in ("group_column", "time_column"):
            if params.get(key) is not None and not isinstance(params[key], str):
                raise bad(f"{key} must be a column name.")
        for key in ("split_mode", "calibration", "missing_policy"):
            if key in params and not isinstance(params[key], str):
                raise bad(f"{key} must be text.")
        if "seed" in params and (isinstance(params["seed"], bool) or not isinstance(params["seed"], int)):
            raise bad("seed must be an integer.")
        if "review_fractions" in params:
            fractions = params["review_fractions"]
            if (not isinstance(fractions, list) or len(fractions) > 10
                    or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in fractions)):
                raise bad("review_fractions must be a short list of numbers.")
        if params.get("costs") is not None and not isinstance(params["costs"], dict):
            raise bad("costs must be an object.")
        try:
            text = json.dumps(params, allow_nan=False)
        except (TypeError, ValueError):
            raise bad("The job request is not valid JSON.") from None
        if len(text) > MAX_REQUEST_BYTES:
            raise bad("The job request is too large.")
        return json.loads(text)

    def _check_heavy_capacity(self) -> None:
        if len(self._heavy_active) + len(self._heavy_queue) >= self.config.heavy_capacity:
            raise JobError("QUEUE_FULL", 429, "The service is busy with another job. Try again shortly.",
                           {"retry_after_seconds": 15})

    # ================================================================ legacy compatibility lanes

    def run_light(self, command: str, payload: dict, timeout: float | None = None) -> dict:
        """Run a short legacy task (``legacy_profile`` or ``legacy_convert``) and return its result untouched."""
        if command not in LIGHT_COMMANDS and not (self.config.allow_test_commands and command in TEST_COMMANDS):
            raise JobError("JOB_COMMAND_UNKNOWN", 400, "Unknown job command.")
        return self._run_light_internal(command, {"payload": payload}, timeout)

    def run_heavy(self, command: str, payload: dict, timeout: float | None = None) -> HeavyHandle:
        """Queue a legacy heavy task. Takes the same one active and one waiting slot as a v1 training job."""
        if command not in HEAVY_COMMANDS or command == "train":
            if not (self.config.allow_test_commands and command in TEST_COMMANDS):
                raise JobError("JOB_COMMAND_UNKNOWN", 400, "Unknown job command.")
        body = {"payload": payload, "model_timeout_seconds": None,
                "job_timeout_seconds": float(timeout or self.config.job_timeout_seconds)}
        with self._lock:
            self._require_running()
            self._check_heavy_capacity()
            job_id = new_id("job")
            directory = self._write_request(self.store.job_dir(job_id), command, body)
            job = _Job(job_id, None, "legacy", "heavy", command, directory, None, self._clock(),
                       float(timeout or self.config.job_timeout_seconds))
            job.reserve = self._reserve_for(command)
            self._jobs[job_id] = job
            self._heavy_queue.append(job)
        self._wake.set()
        return HeavyHandle(self, job)

    def result_cap(self, command: str) -> int:
        """Largest result file, in bytes, a child may leave for this command (legacy commands have their own cap)."""
        return self.config.max_legacy_result_bytes if is_legacy_command(command) else self.config.max_result_bytes

    def _reserve_for(self, command: str) -> int:
        """Storage a job holds back for the result it may still write. Legacy jobs reserve their own cap (a
        profile only its small share, because its result holds statistics only); other light and test
        commands reserve nothing, as before."""
        if not is_legacy_command(command):
            return 0
        cap = self.config.max_legacy_result_bytes
        return min(cap, LEGACY_PROFILE_RESERVE) if command == "legacy_profile" else cap

    def _live_reserved(self) -> int:
        """Caller holds the lock. Result space promised to live jobs, counted on top of what is already on disk
        (a job that has written part of its result is counted twice, which only makes admission stricter)."""
        return sum(job.reserve for job in self._jobs.values() if job.live)

    def _write_request(self, directory: Path, command: str, body: dict) -> Path:
        """Caller holds the lock. Writes ``request.json`` and checks the storage cap first, with the reserve of
        this job and of every live legacy job counted against it."""
        text = json.dumps(dict(body, command=command))
        if (self.store.used_bytes() + self._live_reserved() + self._reserve_for(command) + len(text)
                > self.config.max_stored_bytes):
            raise JobError("STORAGE_FULL", 503, "The service has no room for this request right now.",
                           {"retry_after_seconds": 30})
        self.store.make_dir(directory)
        (directory / "request.json").write_text(text, encoding="utf-8")
        return directory

    def _run_light_internal(self, command: str, body: dict, timeout) -> dict:
        run_timeout = float(timeout or self.config.light_timeout_seconds)
        with self._lock:
            self._require_running()
            occupied = (1 if self._light_active is not None else 0) + len(self._light_queue)
            if occupied >= 1 + self.config.light_max_waiting:
                raise JobError("QUEUE_FULL", 429, "The service is busy. Try again shortly.",
                               {"retry_after_seconds": 5})
            job_id = new_id("light")
            body = dict(body, job_timeout_seconds=run_timeout)
            directory = self._write_request(self.store.light_dir(job_id), command, body)
            job = _Job(job_id, None, "light", "light", command, directory, None, self._clock(), run_timeout)
            job.reserve = self._reserve_for(command)
            self._jobs[job_id] = job
            self._light_queue.append(job)
        self._wake.set()
        wait = self.config.light_wait_seconds if timeout is None else float(timeout) + self.config.light_wait_seconds
        try:
            if not job.done.wait(wait):
                self._cancel(job)
                raise JobError("LIGHT_TIMEOUT", 503, "The request took too long to start. Try again shortly.")
        except BaseException:
            if not job.done.is_set():
                self._cancel(job)
            raise
        return self._legacy_outcome(job)

    def _legacy_outcome(self, job: _Job) -> dict:
        """Result of a light or legacy job, or the matching ``JobError``. The job is dropped once consumed."""
        with self._lock:
            if job.state == states.SUCCEEDED and job.payload is not None:
                payload = job.payload
                self._jobs.pop(job.id, None)
                return payload
            reason = job.reason or {"code": "JOB_FAILED", "message": "The job did not finish."}
            self._jobs.pop(job.id, None)
            state = job.state
            command = job.command
        raise self._error_for(state, reason, command)

    @staticmethod
    def _error_for(state: str, reason: dict, command: str) -> JobError:
        code, message = reason.get("code", "JOB_FAILED"), reason.get("message", "The job did not finish.")
        if state == states.FAILED:
            if code.startswith("LEGACY_"):
                return JobError(code, 400, message)
            if is_legacy_command(command) and code in ("RESULT_TOO_LARGE", MEMORY_LIMIT_REASON["code"]):
                return JobError(code, 400, message)   # the old server answered every refusal with 400
            if code in INTERNAL_CODES:
                return JobError(code, 500, message)
            if code == "JOB_OUT_OF_MEMORY":
                return JobError(code, 503, message)
            return JobError(code, 422, message)
        if state == states.TIMED_OUT:
            return JobError(code, 503, message)
        if state == states.CANCELLED:
            return JobError("CANCELLED", 409, "The job was cancelled.")
        return JobError("SERVICE_STOPPING", 503, "The job was interrupted because the service is stopping.")

    # ================================================================ cancel

    def _cancel(self, job: _Job) -> None:
        """Cancel a job (queued: removed now; running: the supervisor kills the tree and reaps it)."""
        with self._lock:
            if not job.live:
                return
            if job.state == states.QUEUED:
                self._remove_from_queues(job)
                self._finalize(job, states.CANCELLED, {"code": "CANCELLED", "message": "The job was cancelled."})
                return
            job.cancel_requested = True
        self._wake.set()
        job.done.wait(self.config.kill_grace_seconds + 10)

    def _remove_from_queues(self, job: _Job) -> None:
        for queue in (self._heavy_queue, self._light_queue):
            try:
                queue.remove(job)
            except ValueError:
                pass

    # ================================================================ supervisor

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - the supervisor must survive a bad tick
                log.error("supervisor tick failed (%s)", type(exc).__name__)
        try:
            self._shutdown_all()
        except Exception as exc:  # noqa: BLE001
            log.error("supervisor shutdown failed (%s)", type(exc).__name__)

    def _active_jobs(self) -> list:
        jobs = list(self._heavy_active)
        if self._light_active is not None:
            jobs.append(self._light_active)
        return jobs

    def _tick(self) -> None:
        kills, exited = [], []
        with self._lock:
            now = self._clock()
            for job in self._active_jobs():
                if job.process.poll() is not None:
                    exited.append(job)               # a natural exit wins a race with cancel or timeout
                elif job.cancel_requested:
                    kills.append((job, states.CANCELLED, {"code": "CANCELLED", "message": "The job was cancelled."}))
                elif now - job.started_at > job.timeout:
                    kills.append((job, states.TIMED_OUT, {"code": "JOB_TIMEOUT", "message":
                                                          "The job ran longer than the time limit and was stopped."}))
                elif job.process.memory_exceeded():
                    kills.append((job, states.FAILED, dict(MEMORY_LIMIT_REASON)))
            for queue in (self._heavy_queue, self._light_queue):
                for job in list(queue):
                    if job.cancel_requested:
                        queue.remove(job)
                        self._finalize(job, states.CANCELLED,
                                       {"code": "CANCELLED", "message": "The job was cancelled."})
                    elif now - job.queued_at > self._queue_limit(job):
                        queue.remove(job)
                        self._finalize(job, states.TIMED_OUT, {"code": "QUEUE_WAIT_EXCEEDED", "message":
                                       "The job waited too long in the queue and was removed."})
        for job, state, reason in kills:
            reaped = job.process.kill_tree(self.config.kill_grace_seconds)
            if not reaped:
                log.error("a process tree could not be reaped job=%s", _short(job.id))
        with self._lock:
            for job in exited:
                self._collect(job)
            for job, state, reason in kills:
                if job.live:
                    job.process.release()
                    self._finalize(job, state, reason)
            self._start_queued()
            now = self._clock()
            if now >= self._next_sweep:
                self._next_sweep = now + self.config.sweep_interval_seconds
                self._sweep(now)

    def _queue_limit(self, job: _Job) -> float:
        return self.config.queue_wait_seconds if job.lane == "heavy" else self.config.light_wait_seconds

    def _start_queued(self) -> None:
        while self._heavy_queue and len(self._heavy_active) < self.config.max_active_heavy:
            job = self._heavy_queue.popleft()
            if self._launch(job):
                self._heavy_active.append(job)
        if self._light_active is None and self._light_queue:
            job = self._light_queue.popleft()
            if self._launch(job):
                self._light_active = job

    def _launch(self, job: _Job) -> bool:
        try:
            job.process = spawn_runner(self.config, job.command, job.dir, self.runner_path)
        except (OSError, ValueError):
            job.state = states.check_transition(job.state, states.RUNNING)
            self._finalize(job, states.FAILED, {"code": "JOB_START_FAILED",
                                                "message": "The worker process could not be started."})
            return False
        job.started_at = self._clock()
        job.state = states.RUNNING
        log.info("job started id=%s lane=%s", _short(job.id), job.lane)
        return True

    def _collect(self, job: _Job) -> None:
        """The root process exited by itself: read what it reported and map it to a final state."""
        exceeded = job.process.memory_exceeded()   # Windows: read before the job handle is closed below
        job.process.cleanup(self.config.kill_grace_seconds)
        job.exit_code = job.process.returncode
        if exceeded:   # a tree that went past its ceiling fails the job even if the root then reported something
            self._finalize(job, states.FAILED, dict(MEMORY_LIMIT_REASON))
            return
        try:
            error = read_json_bounded(job.dir / "error.json", 16 * 1024)
            result_path = job.dir / "result.json"
            has_result = result_path.exists()
            if isinstance(error, dict):
                self._finalize(job, states.FAILED, self._reported_reason(error))
                return
            if not has_result:
                self._finalize(job, states.FAILED, {"code": "JOB_CRASHED", "message":
                               "The worker process ended without a result."})
                return
            cap = self.result_cap(job.command)
            if result_path.stat().st_size > cap:
                self._finalize(job, states.FAILED, {"code": "RESULT_TOO_LARGE", "message":
                               "The result is larger than the service allows."
                               + (f" The limit is {size_text(cap)}. Try a smaller file."
                                  if is_legacy_command(job.command) else "")})
                return
            if job.kind in ("light", "legacy"):
                job.payload = json.loads(result_path.read_bytes().decode("utf-8"))
                self._finalize(job, states.SUCCEEDED, None)
                return
            report = json.loads(result_path.read_bytes().decode("utf-8"))
        except (StoreError, OSError, ValueError):
            self._finalize(job, states.FAILED, {"code": "JOB_CRASHED", "message":
                           "The worker process ended without a readable result."})
            return
        state, reason, models = self._map_training(report)
        self._finalize(job, state, reason, models=models)

    @staticmethod
    def _reported_reason(error: dict) -> dict:
        code = error.get("code")
        if not isinstance(code, str) or not _CODE.match(code):
            code = "JOB_FAILED"
        message = error.get("message")
        reason = {"code": code, "message": _safe(message, 2000 if code.startswith("LEGACY_") else 300)
                  if isinstance(message, str) else "The job failed."}
        if code.startswith("LEGACY_") and isinstance(message, str):
            reason["message"] = message[:2000]
        return reason

    @staticmethod
    def _map_training(report) -> tuple:
        """Training-1.1 model statuses -> job state: succeeded, partial, failed or timed_out."""
        outcomes = report.get("outcomes") if isinstance(report, dict) else None
        if not isinstance(outcomes, dict) or not outcomes:
            return states.FAILED, {"code": "JOB_RESULT_INVALID", "message":
                                   "The worker returned a result in an unexpected shape."}, []
        models, reasons = [], []
        for name, row in outcomes.items():
            row = row if isinstance(row, dict) else {}
            raw_status = row.get("status")
            status = {"success": "succeeded", "timeout": "timed_out"}.get(raw_status, "failed")
            entry = {"model": _safe(name, 64), "status": status}
            if status != "succeeded":
                code = row.get("reason_code")
                code = code if isinstance(code, str) and _CODE.match(code) else (
                    "MODEL_TIMEOUT" if status == "timed_out" else "MODEL_FIT_ERROR")
                entry["reason_code"] = code
                reasons.append(code)
            models.append(entry)
        succeeded = sum(1 for item in models if item["status"] == "succeeded")
        if succeeded == len(models):
            return states.SUCCEEDED, None, models
        if succeeded:
            return states.PARTIAL, {"code": "SOME_MODELS_FAILED", "message":
                                    f"{len(models) - succeeded} of {len(models)} models did not finish; "
                                    "the report lists each one with its reason."}, models
        if all(code == "MODEL_TIMEOUT" for code in reasons):
            return states.TIMED_OUT, {"code": "MODEL_TIMEOUT", "message":
                                      "Every model reached the per-model time limit."}, models
        code = reasons[0] if len(set(reasons)) == 1 else "ALL_MODELS_FAILED"
        return states.FAILED, {"code": code, "message": "No model could be fitted; the reasons are listed per model."}, models

    def _finalize(self, job: _Job, state: str, reason, models=None) -> None:
        """Caller holds the lock. Moves the job to a terminal state and releases everything it held."""
        states.check_transition(job.state, state)
        now = self._clock()
        if job.state == states.RUNNING and job.started_at is not None:
            job.elapsed = round(now - job.started_at, 3)
        else:
            job.elapsed = 0.0
        if job.state == states.RUNNING and job.kind == "train":
            job.progress = self._read_progress(job)
        job.state = state
        job.finished_at = now
        job.reason = reason
        if models is not None:
            job.models = models
        elif job.progress and job.progress.get("models_completed"):
            job.models = [dict(item) for item in job.progress["models_completed"]]
        if job in self._heavy_active:
            self._heavy_active.remove(job)
        if job is self._light_active:
            self._light_active = None
        if job.process is not None:
            job.process.release()
        if job.kind == "train":
            job.expires_at = now + self.config.finished_job_seconds
            dataset = self._datasets.get(job.dataset_id)
            if dataset is not None:
                dataset.last_used = now
            if state not in TERMINAL_WITH_DIRECTORY:
                self._delete_path(job.dir)
        else:
            self._delete_path(job.dir)
        job.done.set()
        log.info("job finished id=%s state=%s code=%s", _short(job.id), state, (reason or {}).get("code", "-"))

    def _force_interrupt(self, job: _Job) -> None:
        """Last resort in ``stop``: the supervisor did not finish, so settle the job here."""
        if job.live:
            self._remove_from_queues(job)
            self._finalize(job, states.INTERRUPTED, {"code": "SERVICE_STOPPING", "message":
                                                     "The service is stopping."})
        job.done.set()

    def _shutdown_all(self) -> None:
        """Supervisor exit: interrupt every live job, killing the process trees first."""
        with self._lock:
            live = [job for job in self._jobs.values() if job.live]
            running = [job for job in live if job.process is not None and job.process.poll() is None]
        for job in running:
            job.process.kill_tree(self.config.kill_grace_seconds)
        with self._lock:
            for job in live:
                if job.live:
                    self._remove_from_queues(job)
                    self._finalize(job, states.INTERRUPTED, {"code": "SERVICE_STOPPING", "message":
                                                             "The service is stopping."})

    # ================================================================ sweep and storage

    def _delete_path(self, path: Path) -> None:
        if not remove_tree(path, attempts=2, pause=0.1):
            with self._lock:
                self._pending_delete.add(path)

    def _expire_job(self, job: _Job, now: float) -> None:
        states.check_transition(job.state, states.EXPIRED)
        job.state = states.EXPIRED
        self._jobs.pop(job.id, None)
        self._delete_path(job.dir)
        if job.digest is not None:
            self._tombstones[job.id] = ("job", job.digest, now + self.config.tombstone_seconds)
        log.info("job expired id=%s", _short(job.id))

    def _expire_dataset(self, dataset: _Dataset, now: float) -> None:
        self._datasets.pop(dataset.id, None)
        self._delete_path(dataset.dir)
        self._tombstones[dataset.id] = ("dataset", dataset.digest, now + self.config.tombstone_seconds)
        log.info("dataset expired id=%s", _short(dataset.id))

    def _sweep(self, now: float) -> None:
        """Caller holds the lock. Expire by lifetime, retry failed deletions, then enforce the byte cap."""
        for identifier in [key for key, entry in self._tombstones.items() if entry[2] <= now]:
            del self._tombstones[identifier]
        for job in [item for item in self._jobs.values()
                    if item.kind == "train" and item.expires_at is not None and item.expires_at <= now
                    and item.state in states.FINISHED_STATES]:
            self._expire_job(job, now)
        for job in [item for item in self._jobs.values() if item.kind != "train" and item.finished_at is not None
                    and now - item.finished_at > 60.0]:
            self._jobs.pop(job.id, None)   # a legacy or light result nobody collected
        for dataset in [item for item in self._datasets.values() if now - item.last_used >= self.config.dataset_idle_seconds
                        and not any(job.dataset_id == item.id and job.live for job in self._jobs.values())]:
            self._expire_dataset(dataset, now)
        for path in list(self._pending_delete):
            if remove_tree(path, attempts=1, pause=0):
                self._pending_delete.discard(path)
        self._enforce_storage(now)

    def _enforce_storage(self, now: float) -> None:
        cap = self.config.max_stored_bytes
        while self.store.used_bytes() > cap:
            finished = sorted((job for job in self._jobs.values() if job.kind == "train"
                               and job.state in states.RESULT_STATES), key=lambda item: item.finished_at or 0)
            if finished:
                self._expire_job(finished[0], now)
                log.warning("storage cap reached: oldest finished job removed")
                continue
            idle = sorted((item for item in self._datasets.values()
                           if not any(job.dataset_id == item.id and job.live for job in self._jobs.values())),
                          key=lambda item: item.last_used)
            if idle:
                self._expire_dataset(idle[0], now)
                log.warning("storage cap reached: oldest idle dataset removed")
                continue
            break
