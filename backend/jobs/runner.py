"""Child process entry point: ``python backend/jobs/runner.py <command> <job_dir>``.

The coordinator starts this script in its own process group (or job). It reads ``request.json`` from the job
directory, does the work with the services layer or the legacy engine, writes ``result.json`` or
``error.json`` atomically and keeps ``progress.json`` current. It exits 0 for every reported outcome (success
or typed error) and nonzero only when it dies without reporting. Nothing here writes a traceback, a token, a
raw row or a local path into a file the service returns.

A watchdog thread polls the parent process every second. If the parent is gone, this process kills its own
whole tree, so a crashed or killed server leaves no orphan behind.

Commands: ``profile``, ``train``, ``legacy_profile``, ``legacy_analyze``, ``legacy_score``, ``legacy_convert``
and, only when the environment variable ``FYD_ALLOW_TEST_COMMANDS`` is ``1``, ``test_sleep``,
``test_spawn``, ``test_crash``, ``test_big_output``, ``test_env``, ``test_memory`` and ``test_legacy_big_output`` (the last one
answers like a legacy command: legacy result cap, legacy error shape).

Result caps: ``FYD_MAX_RESULT_BYTES`` (default 8 MB) for the v1 commands, ``FYD_MAX_LEGACY_RESULT_BYTES``
(default 32 MB) for the legacy ones. ``FYD_CHILD_MEMORY_MB`` is enforced by the parent (see ``proc.py``), not here.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent

# This directory holds modules named config, store, tokens and so on. Keep them from shadowing anything.
sys.path[:] = [entry for entry in sys.path if entry and Path(entry).resolve() != HERE]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _load(name: str):
    spec = importlib.util.spec_from_file_location("fyd_jobs_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


proc = _load("proc")

WATCHDOG_INTERVAL_SECONDS = 1.0
MAX_DETAILS_BYTES = 4096
MAX_LEGACY_MESSAGE = 2000
LEGACY_KEY_MESSAGE = ("The request is missing a required field, or names a column that is not in the file: {key}.")
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_PATH = re.compile(r"(?:[A-Za-z]:[\\/][^\s'\"<>|]*)|(?:\\\\[^\s'\"<>|]+)|(?:(?<![\w.])/(?:[\w.\-~]+/)+[\w.\-~]*)")
LEGACY_COMMANDS = ("legacy_profile", "legacy_analyze", "legacy_score", "legacy_convert", "legacy_tools")
LEGACY_TEST_COMMANDS = ("test_legacy_big_output",)
TEST_COMMANDS = ("test_sleep", "test_spawn", "test_crash", "test_big_output", "test_env", "test_memory") \
    + LEGACY_TEST_COMMANDS
REAL_COMMANDS = ("profile", "train") + LEGACY_COMMANDS


class Reported(Exception):
    """An outcome to report as ``error.json`` (stable code, safe message)."""

    def __init__(self, code: str, message: str, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}


# ---------------------------------------------------------------------------------------- text safety

_SCRUB_ROOTS: list = []


def _scrub_exact(text: str) -> str:
    """Replace this service's own storage, install and interpreter paths (used on whole result files)."""
    for variant in _SCRUB_ROOTS:
        text = text.replace(variant, "<path>")
    return text


def _scrub(text: str) -> str:
    """Replace the private paths and any file system path with a marker (used on short messages only)."""
    return _PATH.sub("<path>", _scrub_exact(text))


def _safe_line(value, limit: int = 300) -> str:
    text = str(value).strip().splitlines()[0] if str(value).strip() else ""
    text = _scrub(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _scrub_roots(job_dir: Path) -> None:
    candidates = {str(job_dir.resolve().parent.parent), str(job_dir.parent.parent), str(BACKEND.parent), str(BACKEND),
                  sys.prefix, sys.base_prefix}
    try:
        import tempfile
        candidates.add(tempfile.gettempdir())
    except Exception:  # noqa: BLE001
        pass
    variants = set()
    for item in candidates:
        if len(item) < 8:
            continue
        variants.add(item)
        variants.add(item.replace("\\", "\\\\"))
        variants.add(item.replace("\\", "/"))
    _SCRUB_ROOTS[:] = sorted(variants, key=len, reverse=True)


# ---------------------------------------------------------------------------------------- files

def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
    for attempt in range(6):  # Windows refuses to replace a file that a reader has open at that instant
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.02)


def _write_error(job_dir: Path, code: str, message: str, details=None) -> None:
    if not _CODE.match(code or ""):
        code = "JOB_FAILED"
    body = {"code": code, "message": message, "details": details or {}}
    text = json.dumps(body)
    if len(text) > MAX_DETAILS_BYTES + 1000:
        body["details"] = {}
        text = json.dumps(body)
    _write_text_atomic(job_dir / "error.json", text)


class Progress:
    """Writes ``progress.json``. Only stages and counted events; never a percentage derived from time."""

    def __init__(self, job_dir: Path, started: float, limits: dict, models=()):
        self.path = job_dir / "progress.json"
        self.started = started
        self.lock = threading.Lock()
        self.state = {"stage": "parsing", "models_selected": list(models), "models_completed": [],
                      "current_model": None, "elapsed_seconds": 0.0, "limits": limits, "warnings": []}
        self.flush()

    def flush(self) -> None:
        with self.lock:
            self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
            try:
                _write_text_atomic(self.path, json.dumps(self.state))
            except OSError:
                pass

    def stage(self, name: str) -> None:
        self.state["stage"] = name
        self.flush()

    def models(self, ids) -> None:
        self.state["models_selected"] = [str(item) for item in ids][:50]
        self.flush()

    def event(self, event: dict) -> None:
        """Package ``progress`` callback: small JSON-safe dicts (see ``train_comparison_tabular``)."""
        kind = event.get("event")
        if kind == "splitting":
            self.state["stage"] = "splitting"
        elif kind == "model_started":
            self.state["stage"] = "fitting"
            self.state["current_model"] = str(event.get("model"))
        elif kind == "model_finished":
            status = {"success": "succeeded", "timeout": "timed_out"}.get(event.get("status"), "failed")
            entry = {"model": str(event.get("model")), "status": status}
            if event.get("reason_code"):
                entry["reason_code"] = str(event["reason_code"])[:64]
            self.state["models_completed"].append(entry)
            self.state["current_model"] = None
        elif kind == "reporting":
            self.state["stage"] = "reporting"
            self.state["current_model"] = None
        self.flush()

    def done(self) -> None:
        self.state["stage"] = "done"
        self.state["current_model"] = None
        self.flush()


# ---------------------------------------------------------------------------------------- watchdog

def _start_watchdog() -> None:
    raw = os.environ.get("FYD_PARENT_PID", "")
    if not raw.isdigit():
        return
    watch = proc.ParentWatch(int(raw))
    if not watch.alive():
        proc.kill_own_tree()

    def loop():
        while True:
            time.sleep(WATCHDOG_INTERVAL_SECONDS)
            if not watch.alive():
                proc.kill_own_tree()

    threading.Thread(target=loop, name="fyd-parent-watchdog", daemon=True).start()


# ---------------------------------------------------------------------------------------- real commands

def _data_bytes(request: dict, job_dir: Path) -> bytes:
    root = job_dir.resolve().parent.parent
    path = Path(str(request.get("data_file", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise Reported("JOB_REQUEST_INVALID", "The job request names a file outside the job storage.") from None
    return path.read_bytes()


def _limits(request: dict):
    from rare_event_eval.tabular import TabularLimits
    return TabularLimits(**request["limits"])


def _sanitize_training(result: dict) -> dict:
    """Remove worker diagnostics (stderr tails) and make the error texts path free. Metrics are untouched."""
    for row in (result.get("outcomes") or {}).values():
        if not isinstance(row, dict):
            continue
        row.pop("diagnostic", None)
        if isinstance(row.get("error"), str):
            row["error"] = _safe_line(row["error"])
    for item in result.get("failures") or []:
        if isinstance(item, dict) and isinstance(item.get("message"), str):
            item["message"] = _safe_line(item["message"])
    return result


def _cmd_profile(request: dict, job_dir: Path, progress: Progress):
    import services
    progress.stage("parsing")
    data = _data_bytes(request, job_dir)
    progress.stage("profiling")
    record = services.profile_for_app(data, str(request.get("filename") or ""), target=request.get("target"),
                                      limits=_limits(request), import_selection=request.get('import_selection'))
    services.assert_strict_json(record)
    return record, False


def _cmd_train(request: dict, job_dir: Path, progress: Progress):
    import services
    params = dict(request.get("params") or {})
    models = params.get("models")
    progress.models(models if models else services.app_default_models())
    progress.stage("parsing")
    data = _data_bytes(request, job_dir)
    result = services.train_for_app(data, filename=str(request.get("filename") or ""), limits=_limits(request),
                                    timeout_seconds=request["model_timeout_seconds"], progress=progress.event,
                                    import_selection=request.get('import_selection'),
                                    **params)
    return _sanitize_training(result), False


def _legacy_engine():
    import warnings
    warnings.filterwarnings("ignore")
    import engine
    return engine


def _cmd_legacy(command: str, request: dict, job_dir: Path, progress: Progress):
    payload = request.get("payload") or {}
    if command == "legacy_tools":
        from services.toolbox import execute
        progress.stage("fitting")
        return execute(payload), False
    engine = _legacy_engine()
    progress.stage("profiling" if command == "legacy_profile" else "fitting")
    if command == "legacy_profile":
        return engine.profile(engine.read_csv(payload["csv"])), True
    if command == "legacy_analyze":
        target = payload.get("target")
        if not target:
            raise ValueError("Could not read that file well enough to find a target column. "
                             "Check the file's format.")
        folds = max(1, min(100, int(payload.get("folds", 1))))
        return engine.analyze(payload["csv"], target, payload.get("method", "smote"),
                              do_classify=bool(payload.get("do_classify")),
                              classifier=payload.get("classifier", "logreg"), folds=folds), True
    if command == "legacy_score":
        return engine.score_new(payload["train_csv"], payload["target"], payload.get("method", "smote"),
                                payload.get("classifier", "random_forest"), payload["score_csv"]), True
    import base64
    raw = base64.b64decode(payload["b64"])
    if len(raw) > 30 * 1024 * 1024:
        raise ValueError("That file is larger than 30 MB. Please upload a sample "
                         "of the data (the comparison subsamples big data anyway).")
    csv_text, notes = engine.any_to_csv(raw, payload.get("name", ""))
    return {"ok": True, "csv": csv_text, "notes": notes}, True


# ---------------------------------------------------------------------------------------- test commands

def _cmd_test(command: str, request: dict, job_dir: Path, progress: Progress):
    if command == "test_sleep":
        deadline = time.monotonic() + float(request.get("seconds", 1))
        progress.stage("fitting")
        while time.monotonic() < deadline:
            time.sleep(0.05)
        return {"slept": request.get("seconds", 1)}, False
    if command == "test_spawn":
        import subprocess
        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", str(job_dir)],
                                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, creationflags=proc.CREATE_NO_WINDOW)
        _write_text_atomic(Path(request["pid_file"]), json.dumps({"runner": os.getpid(), "grandchild": grandchild.pid}))
        progress.stage("fitting")
        deadline = time.monotonic() + float(request.get("seconds", 60))
        while time.monotonic() < deadline:
            time.sleep(0.05)
        return {"spawned": True}, False
    if command == "test_crash":
        mode = request.get("mode", "hard_exit")
        if mode == "hard_exit":
            os._exit(3)
        if mode == "exception":
            raise RuntimeError("boom at " + str(request.get("leak_path", "C:\\secret\\place\\module.py"))
                               + " token=" + str(request.get("leak_token", "")))
        if mode == "memory":
            raise MemoryError("simulated")
        raise Reported("TEST_REPORTED", "A reported test failure.")
    if command in ("test_big_output", "test_legacy_big_output"):
        size = int(request.get("bytes", 20_000_000))
        legacy_shape = command == "test_legacy_big_output"
        delay = float(request.get("delay", 0))
        if delay > 0:
            progress.stage("fitting")
            time.sleep(delay)
        if request.get("bypass"):  # write straight to result.json, ignoring the runner's own cap
            (job_dir / "result.json").write_bytes(b'{"blob":"' + b"x" * size + b'"}')
            return None, False
        return ({"ok": True, "blob": "x" * size} if legacy_shape else {"blob": "x" * size}), legacy_shape
    if command == "test_memory":
        import subprocess
        step = max(1, int(request.get("step_megabytes", 20))) * 1024 * 1024
        steps = max(1, int(request.get("megabytes", 400)) * 1024 * 1024 // step)
        hold = float(request.get("hold_seconds", 30))
        progress.stage("fitting")
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", str(job_dir)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=proc.CREATE_NO_WINDOW)
        allocator = None
        if request.get("allocator") == "grandchild":   # the memory is taken by a descendant, not by this process
            code = ("import sys, time\nblocks = []\nfor _ in range(int(sys.argv[1])):\n"
                    "    blocks.append(bytearray(b'x') * int(sys.argv[2]))\n    time.sleep(0.1)\n"
                    "time.sleep(float(sys.argv[3]))")
            allocator = subprocess.Popen([sys.executable, "-c", code, str(steps), str(step), str(hold)],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, creationflags=proc.CREATE_NO_WINDOW)
        if request.get("pid_file"):
            _write_text_atomic(Path(request["pid_file"]), json.dumps(
                {"runner": os.getpid(), "grandchild": sleeper.pid, "allocator": allocator.pid if allocator else None}))
        blocks = []
        if allocator is None:
            for _ in range(steps):
                blocks.append(bytearray(b"x") * step)   # a repeated byte, so every page is really used
                time.sleep(0.1)
        deadline = time.monotonic() + hold
        while time.monotonic() < deadline:
            time.sleep(0.05)
        return {"allocated_bytes": len(blocks) * step}, False
    if command == "test_env":
        names = [str(item) for item in request.get("names", [])]
        keys = [key for key in os.environ if any(name.lower() in key.lower() for name in names)]
        return {"visible": sorted(keys), "threads": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT")},
            "temp_inside_job": str(Path(os.environ.get("TMP", "")).resolve()).startswith(str(job_dir.resolve())),
            "cwd_is_job": Path.cwd().resolve() == job_dir.resolve()}, False
    raise Reported("JOB_COMMAND_UNKNOWN", "Unknown job command.")


# ---------------------------------------------------------------------------------------- main

def _classify(exc: BaseException, legacy: bool):
    """(code, message, details) for an exception. Never a traceback, never a path."""
    if isinstance(exc, Reported):
        return exc.code, exc.message, exc.details
    if legacy:
        if isinstance(exc, KeyError):
            return "LEGACY_BAD_REQUEST", LEGACY_KEY_MESSAGE.format(key=exc), {}
        if isinstance(exc, MemoryError):
            return "JOB_OUT_OF_MEMORY", "The service ran out of memory for this request.", {}
        return "LEGACY_ERROR", _scrub(str(exc))[:MAX_LEGACY_MESSAGE], {}
    code = getattr(exc, "code", None)
    if isinstance(code, str) and _CODE.match(code):
        details = getattr(exc, "details", None)
        try:
            details = json.loads(json.dumps(details, allow_nan=False, default=str)) if details else {}
        except (TypeError, ValueError):
            details = {}
        if len(json.dumps(details)) > MAX_DETAILS_BYTES:
            details = {}
        return code, _safe_line(exc), details
    if isinstance(exc, MemoryError):
        return "JOB_OUT_OF_MEMORY", "The service ran out of memory for this request.", {}
    return ("JOB_INTERNAL_ERROR",
            f"The job failed unexpectedly ({type(exc).__name__}). No further details are shown.", {})


def main(argv) -> int:
    if len(argv) != 3:
        return 2
    command, job_dir = argv[1], Path(argv[2])
    started = time.monotonic()
    _start_watchdog()
    proc.adopt_self_into_job()
    _scrub_roots(job_dir)
    legacy = command in LEGACY_COMMANDS or command in LEGACY_TEST_COMMANDS
    max_result = int(os.environ.get("FYD_MAX_LEGACY_RESULT_BYTES", "33554432") if legacy
                     else os.environ.get("FYD_MAX_RESULT_BYTES", "8388608"))
    try:
        request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
        limits = {"model_timeout_seconds": request.get("model_timeout_seconds"),
                  "job_timeout_seconds": request.get("job_timeout_seconds")}
        progress = Progress(job_dir, started, limits)
        if command in TEST_COMMANDS:
            if os.environ.get("FYD_ALLOW_TEST_COMMANDS") != "1":
                raise Reported("JOB_COMMAND_UNKNOWN", "Unknown job command.")
            payload, allow_nan = _cmd_test(command, request.get("payload") or request, job_dir, progress)
        elif command == "profile":
            payload, allow_nan = _cmd_profile(request, job_dir, progress)
        elif command == "train":
            payload, allow_nan = _cmd_train(request, job_dir, progress)
        elif legacy:
            payload, allow_nan = _cmd_legacy(command, request, job_dir, progress)
        else:
            raise Reported("JOB_COMMAND_UNKNOWN", "Unknown job command.")
        if payload is None:  # test_big_output with bypass wrote result.json itself
            return 0
        try:
            text = json.dumps(payload, allow_nan=allow_nan)
        except (TypeError, ValueError):
            raise Reported("APP_REPORT_NOT_JSON_SAFE", "The result holds a value that is not valid JSON.") from None
        if len(text) > max_result:
            message = "The result is larger than the service allows."
            if legacy:
                if max_result < 1024 * 1024:
                    size = f"{max(1, max_result // 1024)} KB"
                else:
                    size = f"{max_result / (1024 * 1024):.1f}"
                    size = (size[:-2] if size.endswith(".0") else size) + " MB"
                message += f" The limit is {size}. Try a smaller file."
            raise Reported("RESULT_TOO_LARGE", message, {"limit_bytes": max_result})
        _write_text_atomic(job_dir / "result.json", text if legacy else _scrub_exact(text))
        progress.done()
        return 0
    except BaseException as exc:  # noqa: BLE001 - this is the reporting boundary of the child process
        if isinstance(exc, KeyboardInterrupt):
            raise
        code, message, details = _classify(exc, legacy)
        try:
            _write_error(job_dir, code, message, details)
        except OSError:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
