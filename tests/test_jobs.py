"""Behaviour tests for the bounded job coordinator (``backend/jobs``).

Almost every test starts real child processes. Most use a small stand-in runner (written to a temporary file
at the start of the module) that imports the real ``backend/jobs/runner.py`` and replaces only the ``train``
and ``profile`` work with controllable behaviour, so the coordinator, the process lifecycle, the files and the
timing are all real while no model is trained. The classes named ``Real...`` use the real runner: a real
profile, real training jobs on a synthetic mixed-type table of about 300 rows, a real model timeout, and the
legacy commands compared with the legacy engine called in this process.

At the end of the module (``tearDownModule``) every process whose command line mentions this run's private
directory is searched for and the temporary directories are checked, so a leaked process or directory fails
the suite.

Run from the repository root::

    python -B -m unittest discover -s tests -p "test_jobs.py" -v
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
PACKAGE_SOURCE = ROOT / "evaluation" / "src"
PACKAGE_TESTS = ROOT / "evaluation" / "tests"
REAL_RUNNER = BACKEND / "jobs" / "runner.py"

if importlib.util.find_spec("rare_event_eval") is None:  # child processes inherit PYTHONPATH
    sys.path.insert(0, str(PACKAGE_SOURCE))
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE_SOURCE), os.environ.get("PYTHONPATH", "")]))
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(PACKAGE_TESTS))

from jobs import Coordinator, HeavyHandle, JobConfig, JobError, states  # noqa: E402
from jobs import proc, store, tokens  # noqa: E402
from jobs.coordinator import child_environment, spawn_runner  # noqa: E402
from mixed_support import mixed_table, table_to_csv  # noqa: E402

try:
    import pandas  # noqa: F401
    import engine as legacy_engine
    LEGACY_REASON = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    legacy_engine = None
    LEGACY_REASON = f"the legacy engine is not importable here ({type(exc).__name__})"

MODULE_BASE: Path
MARKER: str
FAKE_RUNNER: Path
PARENT_HELPER: Path
LEAKED_ROOTS: list = []
SAFETY_NET_KILLS: list = []

FAKE_RUNNER_SOURCE = '''
import importlib.util, json, os, subprocess, sys, time
from pathlib import Path

spec = importlib.util.spec_from_file_location("fyd_real_runner", {real!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fake_profile(request, job_dir, progress):
    progress.stage("profiling")
    return {{"schema": "app-profile-1.0", "rows": 3, "target": {{"column": "label"}}}}, False


def fake_train(request, job_dir, progress):
    params = request["params"]
    kind, _, arg = params["target"].partition(":")
    models = params.get("models") or ["m1"]
    progress.models(models)
    progress.stage("parsing")
    statuses = {{name: "success" for name in models}}
    if kind == "sleep":
        time.sleep(float(arg))
    elif kind in ("spawn", "straggler"):
        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", str(job_dir)],
                                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, creationflags=runner.proc.CREATE_NO_WINDOW)
        Path(arg).write_text(json.dumps({{"runner": os.getpid(), "grandchild": grandchild.pid}}))
        if kind == "spawn":
            time.sleep(120)
    elif kind == "alloc":   # take memory in 20 MB steps, then wait: a worker that outgrows its ceiling
        blocks = []
        for _ in range(int(arg)):
            blocks.append(bytearray(b"x") * (20 * 1024 * 1024))
            time.sleep(0.1)
        time.sleep(60)
    elif kind == "crash":
        if arg == "hard":
            os._exit(3)
        raise RuntimeError("boom at C:\\\\secret\\\\place\\\\x.py Traceback (most recent call last) token=" + arg)
    elif kind == "statuses":
        statuses = dict(item.split("=") for item in arg.split(","))
    elif kind == "steps":
        step = float(arg)
        progress.event({{"event": "splitting"}})
        time.sleep(step)
        for index, name in enumerate(models, start=1):
            progress.event({{"event": "model_started", "model": name, "index": index, "total": len(models)}})
            time.sleep(step)
            progress.event({{"event": "model_finished", "model": name, "index": index, "total": len(models),
                            "status": "success"}})
            time.sleep(step)
        progress.event({{"event": "reporting"}})
        time.sleep(step)
    outcomes = {{}}
    for name, status in statuses.items():
        row = {{"status": status, "model": name}}
        if status != "success":
            row["reason_code"] = "MODEL_TIMEOUT" if status == "timeout" else "MODEL_FIT_ERROR"
        outcomes[name] = row
    return {{"schema_version": "training-1.1", "outcomes": outcomes, "predictors": ["a"],
            "input_sha256": "0" * 64, "interpretation": "stand-in record"}}, False


runner._cmd_profile = fake_profile
runner._cmd_train = fake_train
sys.exit(runner.main(sys.argv))
'''

PARENT_HELPER_SOURCE = '''
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, {backend!r})
from jobs.config import JobConfig
from jobs.coordinator import spawn_runner

base = Path(sys.argv[1])
use_job = sys.argv[2] == "1"
job = base / "job"
job.mkdir()
(job / "request.json").write_text(json.dumps({{"pid_file": str(base / "pids.json"), "seconds": 120}}))
config = JobConfig(allow_test_commands=True, temp_root=str(base / "root"))
child = spawn_runner(config, "test_spawn", job, use_job=use_job)
(base / "helper.pid").write_text(str(os.getpid()))
time.sleep(600)
'''


def setUpModule():
    global MODULE_BASE, MARKER, FAKE_RUNNER, PARENT_HELPER
    MODULE_BASE = Path(tempfile.mkdtemp(prefix="fyd-test-")).resolve()
    MARKER = MODULE_BASE.name
    FAKE_RUNNER = MODULE_BASE / "fake_runner.py"
    FAKE_RUNNER.write_text(FAKE_RUNNER_SOURCE.format(real=str(REAL_RUNNER)), encoding="utf-8")
    PARENT_HELPER = MODULE_BASE / "parent_helper.py"
    PARENT_HELPER.write_text(PARENT_HELPER_SOURCE.format(backend=str(BACKEND)), encoding="utf-8")


def processes_with(marker: str) -> list:
    """Process ids (other than this one) whose command line mentions ``marker``."""
    if os.name == "nt":
        script = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*" + marker + "*' -and "
                  "$_.ProcessId -ne $PID } | ForEach-Object { $_.ProcessId }")
        output = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=120).stdout
        return [int(item) for item in output.split() if item.isdigit() and int(item) != os.getpid()]
    output = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=60).stdout
    found = []
    for line in output.splitlines():
        pid, _, args = line.strip().partition(" ")
        if marker in args and pid.isdigit() and int(pid) != os.getpid():
            found.append(int(pid))
    return found


def kill_only(pid: int) -> None:
    """End one process but not its descendants (taskkill /T would take the whole tree)."""
    try:
        os.kill(pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
    except OSError:
        pass


def kill_pid(pid: int) -> None:
    """End a process and everything it started."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def tearDownModule():
    leftover = processes_with(MARKER)
    for pid in leftover:
        kill_pid(pid)
    roots = [str(path) for path in LEAKED_ROOTS if Path(path).exists()]
    store.remove_tree(MODULE_BASE)
    problems = []
    if leftover:
        problems.append(f"{len(leftover)} process(es) still running after the suite")
    if SAFETY_NET_KILLS:
        problems.append(f"{len(SAFETY_NET_KILLS)} process(es) had to be killed by a test cleanup")
    if roots:
        problems.append(f"{len(roots)} job storage root(s) not deleted by stop()")
    if MODULE_BASE.exists():
        problems.append("the private test directory could not be removed")
    if problems:
        raise AssertionError("; ".join(problems))


def wait_for(predicate, timeout: float = 30.0, step: float = 0.05, message: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    raise AssertionError(f"timed out waiting for {message}")


def error_key(error: JobError) -> tuple:
    return error.code, error.http_status, error.message, json.dumps(error.details, sort_keys=True)


class CoordinatorCase(unittest.TestCase):
    """Starts coordinators with a private storage root and always stops them and removes the directories."""

    def make(self, fake: bool = True, **overrides) -> Coordinator:
        base = Path(tempfile.mkdtemp(prefix="case-", dir=MODULE_BASE))
        defaults = dict(temp_root=str(base / "root"), allow_test_commands=True, sweep_interval_seconds=0.2)
        coordinator = Coordinator(JobConfig(**{**defaults, **overrides}), runner_path=FAKE_RUNNER if fake else None)
        coordinator.start()
        self.addCleanup(self._close, coordinator, base)
        return coordinator

    def _close(self, coordinator: Coordinator, base: Path) -> None:
        coordinator.stop(10)
        if coordinator.store.root.exists():
            LEAKED_ROOTS.append(coordinator.store.root)
        store.remove_tree(base)

    def track(self, pid: int) -> int:
        """Register a pid that must be gone by the end of the test; a cleanup kills it if it is not."""
        def safety_net():
            if proc.pid_alive(pid):
                SAFETY_NET_KILLS.append(pid)
                kill_pid(pid)
        self.addCleanup(safety_net)
        return pid

    def assertDead(self, pid: int, timeout: float = 20.0) -> None:
        wait_for(lambda: not proc.pid_alive(pid), timeout, message=f"process {pid} to exit")

    # ------------------------------------------------------------ helpers
    def dataset(self, coordinator: Coordinator, name: str = "table.csv", raw: bytes = b"a,b\n1,2\n3,4\n") -> dict:
        return coordinator.create_dataset(raw, name)

    @staticmethod
    def request(target: str, **extra) -> dict:
        return {"target": target, "positive_label": "yes", "negative_label": "no", **extra}

    def submit(self, coordinator: Coordinator, dataset: dict, target: str, **extra) -> dict:
        return coordinator.submit_job(dataset["dataset_id"], dataset["capability"], self.request(target, **extra))

    def status(self, coordinator, job) -> dict:
        return coordinator.job_status(job["job_id"], job["capability"])

    def wait_state(self, coordinator, job, wanted, timeout: float = 40.0) -> dict:
        wanted = (wanted,) if isinstance(wanted, str) else tuple(wanted)
        holder = {}

        def reached():
            holder["status"] = self.status(coordinator, job)
            return holder["status"]["state"] in wanted
        wait_for(reached, timeout, message=f"job state {wanted}")
        return holder["status"]

    def pid_file(self, name: str = "pids.json") -> Path:
        return MODULE_BASE / f"{name}.{time.monotonic_ns()}"

    def read_pids(self, path: Path) -> dict:
        wait_for(path.exists, 30, message="the child to write its pids")
        pids = json.loads(wait_for(lambda: path.read_text(encoding="utf-8") or None, 10))
        self.track(pids["runner"])
        self.track(pids["grandchild"])
        return pids


# ============================================================================== units without processes

class StateMachine(unittest.TestCase):
    def test_every_legal_edge_and_no_other(self):
        legal = {
            ("queued", "running"), ("queued", "cancelled"), ("queued", "timed_out"), ("queued", "interrupted"),
            ("running", "succeeded"), ("running", "partial"), ("running", "failed"), ("running", "cancelled"),
            ("running", "timed_out"), ("running", "interrupted"),
            *[(name, "expired") for name in ("succeeded", "partial", "failed", "cancelled", "timed_out", "interrupted")],
        }
        for current in states.ALL_STATES:
            for new in states.ALL_STATES:
                with self.subTest(edge=f"{current}->{new}"):
                    self.assertEqual(states.can_transition(current, new), (current, new) in legal)
                    if (current, new) in legal:
                        self.assertEqual(states.check_transition(current, new), new)
                    else:
                        with self.assertRaises(states.IllegalTransition):
                            states.check_transition(current, new)

    def test_terminal_states_and_expiry_only_from_terminal(self):
        self.assertEqual(states.FINISHED_STATES, {"succeeded", "partial", "failed", "cancelled", "timed_out", "interrupted"})
        self.assertFalse(states.can_transition("queued", "expired"))
        self.assertFalse(states.can_transition("running", "expired"))
        for name in states.FINISHED_STATES:
            self.assertTrue(states.is_terminal(name))
            self.assertFalse(states.can_transition(name, "running"))
            self.assertFalse(states.can_transition(name, "queued"))
        self.assertTrue(states.is_terminal("expired"))
        self.assertFalse(states.is_terminal("queued") or states.is_terminal("running"))


class Capabilities(unittest.TestCase):
    def test_tokens_are_random_url_safe_and_checked_in_constant_time_form(self):
        first, digest = tokens.new_capability()
        second, _ = tokens.new_capability()
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^[A-Za-z0-9_-]{43}$")
        self.assertEqual(digest, tokens.digest_of(first))
        self.assertNotIn(first, digest)
        self.assertTrue(tokens.check_capability(first, digest))
        for bad in (second, "", None, 7, b"bytes", "x" * 1000, first + "x", first[:-1], "\ud800"):
            self.assertFalse(tokens.check_capability(bad, digest), repr(bad)[:20])
        self.assertFalse(tokens.check_capability(first, None))

    def test_identifiers_are_opaque_and_unique(self):
        ids = {tokens.new_id("job") for _ in range(200)}
        self.assertEqual(len(ids), 200)
        self.assertTrue(all(item.startswith("job_") and len(item) > 20 for item in ids))


class ConfigAndStore(unittest.TestCase):
    def test_environment_overrides_and_ignored_bad_values(self):
        config = JobConfig.from_env({"FYD_MAX_DATASETS": "7", "FYD_MODEL_TIMEOUT_SECONDS": "12.5",
                                     "FYD_UPLOAD_LIMIT_BYTES": "not a number", "FYD_ALLOW_TEST_COMMANDS": "1"})
        self.assertEqual((config.max_datasets, config.model_timeout_seconds, config.allow_test_commands), (7, 12.5, True))
        self.assertEqual(config.upload_limit_bytes, JobConfig().upload_limit_bytes)
        self.assertFalse(JobConfig().allow_test_commands)
        with self.assertRaises(ValueError):
            JobConfig(model_timeout_seconds=601)
        with self.assertRaises(ValueError):
            JobConfig(max_datasets=0)

    def test_defaults_match_the_design(self):
        config = JobConfig()
        self.assertEqual((config.max_active_heavy, config.max_waiting_heavy, config.max_jobs_per_dataset,
                          config.max_datasets, config.max_models), (1, 1, 1, 4, 5))
        self.assertEqual((config.upload_limit_bytes, config.max_stored_bytes), (8 * 1024 * 1024, 64 * 1024 * 1024))
        self.assertEqual((config.model_timeout_seconds, config.job_timeout_seconds, config.queue_wait_seconds,
                          config.light_timeout_seconds, config.light_max_waiting), (60, 300, 120, 20, 2))
        self.assertEqual((config.dataset_idle_seconds, config.finished_job_seconds, config.tombstone_seconds),
                         (1800, 1800, 86400))
        self.assertEqual(config.tabular_limits_kwargs(), {"max_bytes": 8 * 1024 * 1024, "max_rows": 10000,
                                                          "max_expanded_features": 100})

    def test_the_legacy_result_cap_and_the_child_memory_ceiling_are_configurable(self):
        config = JobConfig()
        self.assertEqual((config.max_result_bytes, config.max_legacy_result_bytes), (8 * 1024 * 1024, 32 * 1024 * 1024))
        self.assertEqual((config.child_memory_mb, config.child_memory_bytes), (700, 700 * 1024 * 1024))
        config = JobConfig.from_env({"FYD_MAX_LEGACY_RESULT_BYTES": "1048576", "FYD_CHILD_MEMORY_MB": "256"})
        self.assertEqual((config.max_legacy_result_bytes, config.child_memory_mb), (1048576, 256))
        self.assertEqual(JobConfig.from_env({"FYD_CHILD_MEMORY_MB": "0"}).child_memory_bytes, 0)   # 0 turns it off
        self.assertEqual(JobConfig.from_env({"FYD_CHILD_MEMORY_MB": "lots"}).child_memory_mb, 700)  # ignored
        for bad in ({"child_memory_mb": -1}, {"child_memory_mb": True}, {"child_memory_mb": 1.5},
                    {"max_legacy_result_bytes": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                JobConfig(**bad)
        env = child_environment(JobConfig(max_legacy_result_bytes=456, max_result_bytes=123, child_memory_mb=0))
        self.assertEqual((env["FYD_MAX_LEGACY_RESULT_BYTES"], env["FYD_MAX_RESULT_BYTES"], env["FYD_CHILD_MEMORY_MB"]),
                         ("456", "123", "0"))

    def test_atomic_write_and_bounded_read(self):
        with tempfile.TemporaryDirectory(dir=MODULE_BASE) as directory:
            path = Path(directory) / "x.json"
            store.write_json_atomic(path, {"a": 1})
            self.assertEqual(store.read_json_bounded(path, 100), {"a": 1})
            self.assertFalse(Path(str(path) + ".tmp").exists())
            self.assertIsNone(store.read_json_bounded(Path(directory) / "missing.json", 100))
            with self.assertRaises(store.StoreError):
                store.read_json_bounded(path, 3)
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(store.StoreError):
                store.read_json_bounded(path, 100)
            with self.assertRaises(ValueError):
                store.write_json_atomic(path, {"a": float("nan")})

    def test_a_non_empty_directory_without_the_marker_is_never_wiped(self):
        with tempfile.TemporaryDirectory(dir=MODULE_BASE) as directory:
            foreign = Path(directory) / "somebody-elses-folder"
            foreign.mkdir()
            (foreign / "precious.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(store.StoreError):
                store.Store(foreign).prepare_root()
            self.assertEqual((foreign / "precious.txt").read_text(encoding="utf-8"), "keep")
            coordinator = Coordinator(JobConfig(temp_root=str(foreign)))
            with self.assertRaises(RuntimeError):
                coordinator.start()
            self.assertEqual((foreign / "precious.txt").read_text(encoding="utf-8"), "keep")


# ============================================================================== lifecycle and limits

class Startup(CoordinatorCase):
    def test_start_wipes_a_pre_seeded_stale_root(self):
        base = Path(tempfile.mkdtemp(prefix="stale-", dir=MODULE_BASE))
        self.addCleanup(store.remove_tree, base)
        stale = store.Store(base / "root")
        stale.prepare_root()
        (stale.jobs / "job_old").mkdir()
        (stale.jobs / "job_old" / "result.json").write_text("{}", encoding="utf-8")
        (stale.datasets / "ds_old").mkdir()
        (stale.datasets / "ds_old" / "data.bin").write_bytes(b"x" * 1000)
        (stale.light / "light_old").mkdir()
        (base / "root" / "stray.txt").write_text("leftover", encoding="utf-8")
        coordinator = Coordinator(JobConfig(temp_root=str(base / "root"), allow_test_commands=True), runner_path=FAKE_RUNNER)
        coordinator.start()
        self.addCleanup(self._close, coordinator, base)
        self.assertEqual(coordinator.store.leftover_entries(), [])
        self.assertFalse((base / "root" / "stray.txt").exists())
        self.assertEqual(coordinator.stats()["datasets"], 0)
        self.assertEqual(coordinator.stats()["jobs"], 0)
        # An id from the previous process is simply unknown.
        with self.assertRaises(JobError) as caught:
            coordinator.job_status("job_old", "any")
        self.assertEqual(caught.exception.code, "NOT_FOUND")

    def test_stop_interrupts_a_running_job_kills_its_tree_and_removes_the_root(self):
        coordinator = self.make()
        pids_path = self.pid_file()
        handle = coordinator.run_heavy("test_spawn", {"pid_file": str(pids_path), "seconds": 120})
        pids = self.read_pids(pids_path)
        waiting = coordinator.run_heavy("test_sleep", {"seconds": 30})
        root = coordinator.store.root
        coordinator.stop(10)
        self.assertEqual((handle.state, waiting.state), ("interrupted", "interrupted"))
        self.assertDead(pids["runner"])
        self.assertDead(pids["grandchild"])
        self.assertFalse(root.exists())
        with self.assertRaises(JobError) as caught:
            coordinator.run_light("test_sleep", {"seconds": 0})
        self.assertEqual(caught.exception.code, "SERVICE_UNAVAILABLE")
        with self.assertRaises(JobError) as caught:
            handle.wait(1)
        self.assertEqual(caught.exception.code, "SERVICE_STOPPING")
        coordinator.stop(1)  # a second stop is harmless

    def test_children_get_thread_limits_a_private_temp_dir_and_no_secret_looking_variables(self):
        coordinator = self.make()
        with mock.patch.dict(os.environ, {"STATS_KEY": "do-not-pass-this-on", "MY_API_TOKEN": "x", "DB_PASSWORD": "y",
                                          "HARMLESS_SETTING": "kept"}):
            result = coordinator.run_light("test_env", {"names": ["STATS_KEY", "TOKEN", "PASSWORD", "HARMLESS"]})
        self.assertEqual(result["visible"], ["HARMLESS_SETTING"])
        self.assertEqual(set(result["threads"].values()), {"1"})
        self.assertEqual(set(result["threads"]), {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT"})
        self.assertTrue(result["temp_inside_job"])
        self.assertTrue(result["cwd_is_job"])

    def test_test_commands_are_refused_unless_enabled(self):
        coordinator = self.make(allow_test_commands=False)
        for call in (lambda: coordinator.run_light("test_sleep", {"seconds": 0}),
                     lambda: coordinator.run_heavy("test_sleep", {"seconds": 0}),
                     lambda: coordinator.run_light("not_a_command", {}),
                     lambda: coordinator.run_heavy("train", {})):
            with self.assertRaises(JobError) as caught:
                call()
            self.assertEqual((caught.exception.code, caught.exception.http_status), ("JOB_COMMAND_UNKNOWN", 400))
        # The child refuses them too, even if a caller reaches it directly.
        base = Path(tempfile.mkdtemp(prefix="direct-", dir=MODULE_BASE))
        self.addCleanup(store.remove_tree, base)
        job = base / "root" / "jobs" / "j1"
        job.mkdir(parents=True)
        (job / "request.json").write_text(json.dumps({"seconds": 0}), encoding="utf-8")
        child = spawn_runner(JobConfig(allow_test_commands=False), "test_sleep", job, runner_path=FAKE_RUNNER)
        self.track(child.pid)
        wait_for(lambda: child.poll() is not None, 30, message="the child to exit")
        child.cleanup()
        self.assertEqual(json.loads((job / "error.json").read_text(encoding="utf-8"))["code"], "JOB_COMMAND_UNKNOWN")


class Admission(CoordinatorCase):
    def test_a_full_queue_answers_429_at_once_and_legacy_and_v1_jobs_share_the_slots(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        first = coordinator.run_heavy("test_sleep", {"seconds": 60})
        wait_for(lambda: first.state == "running", message="the first job to run")
        second = coordinator.run_heavy("test_sleep", {"seconds": 60})
        self.assertEqual(second.state, "queued")
        for attempt in (lambda: coordinator.run_heavy("test_sleep", {"seconds": 1}),
                        lambda: self.submit(coordinator, dataset, "sleep:1")):
            started = time.monotonic()
            with self.assertRaises(JobError) as caught:
                attempt()
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertEqual((caught.exception.code, caught.exception.http_status), ("QUEUE_FULL", 429))
            self.assertIn("retry_after_seconds", caught.exception.details)
        second.cancel()
        self.assertEqual(second.state, "cancelled")
        accepted = self.submit(coordinator, dataset, "sleep:1")  # the waiting slot is free again
        self.assertEqual(accepted["state"], "queued")
        self.assertEqual(self.status(coordinator, accepted)["queue_position"], 1)
        coordinator.cancel_job(accepted["job_id"], accepted["capability"])
        first.cancel()
        self.assertEqual(first.state, "cancelled")
        self.assertEqual(coordinator.stats()["heavy_running"] + coordinator.stats()["heavy_waiting"], 0)

    def test_a_second_job_on_the_same_dataset_is_rejected_until_the_first_ends(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        first = self.submit(coordinator, dataset, "sleep:60")
        with self.assertRaises(JobError) as caught:
            self.submit(coordinator, dataset, "sleep:1")
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("DATASET_BUSY", 429))
        coordinator.cancel_job(first["job_id"], first["capability"])
        second = self.submit(coordinator, dataset, "sleep:0.1")
        self.assertEqual(self.wait_state(coordinator, second, "succeeded")["state"], "succeeded")

    def test_the_light_lane_runs_one_at_a_time_and_refuses_a_fourth_caller(self):
        coordinator = self.make(light_max_waiting=2)
        results, errors = [], []

        def call():
            try:
                results.append(coordinator.run_light("test_sleep", {"seconds": 1.0}))
            except JobError as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        threads = [threading.Thread(target=call) for _ in range(3)]
        for thread in threads:
            thread.start()
            time.sleep(0.05)
        wait_for(lambda: coordinator.stats()["light_waiting"] == 2 and coordinator.stats()["light_running"] == 1,
                 message="one running and two waiting light tasks")
        started = time.monotonic()
        with self.assertRaises(JobError) as caught:
            coordinator.run_light("test_sleep", {"seconds": 1.0})
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("QUEUE_FULL", 429))
        for thread in threads:
            thread.join(60)
        self.assertEqual((len(results), errors), (3, []))

    def test_at_most_max_datasets_are_kept_and_a_rejected_upload_leaves_nothing_behind(self):
        coordinator = self.make(max_datasets=2)
        first, second = self.dataset(coordinator), self.dataset(coordinator)
        with self.assertRaises(JobError) as caught:
            self.dataset(coordinator)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("DATASET_LIMIT_REACHED", 429))
        self.assertEqual(len(coordinator.store.leftover_entries()), 2)
        coordinator.delete_dataset(first["dataset_id"], first["capability"])
        self.assertEqual(len(coordinator.store.leftover_entries()), 1)
        self.dataset(coordinator)
        self.assertNotEqual(first["dataset_id"], second["dataset_id"])

    def test_upload_limits_and_request_validation(self):
        coordinator = self.make(upload_limit_bytes=1000)
        with self.assertRaises(JobError) as caught:
            coordinator.create_dataset(b"x" * 1001, "big.csv")
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("UPLOAD_TOO_LARGE", 413))
        with self.assertRaises(JobError) as caught:
            coordinator.create_dataset(b"   \n", "blank.csv")
        self.assertEqual(caught.exception.code, "UPLOAD_EMPTY")
        dataset = self.dataset(coordinator)
        bad_requests = {
            "unknown field": ({**self.request("sleep:1"), "surprise": 1}, "REQUEST_INVALID"),
            "no target": ({"positive_label": "yes", "negative_label": "no"}, "REQUEST_INVALID"),
            "no positive": ({"target": "t", "negative_label": "no"}, "REQUEST_INVALID"),
            "neither label mode": ({"target": "t", "positive_label": "yes"}, "REQUEST_INVALID"),
            "both label modes": ({**self.request("t"), "one_vs_rest": True}, "REQUEST_INVALID"),
            "six models": (self.request("t", models=list("abcdef")), "TOO_MANY_MODELS"),
            "duplicate models": (self.request("t", models=["a", "a"]), "REQUEST_INVALID"),
            "empty models": (self.request("t", models=[]), "REQUEST_INVALID"),
            "seed type": (self.request("t", seed="7"), "REQUEST_INVALID"),
            "not an object": ([1, 2], "REQUEST_INVALID"),
            "non finite": (self.request("t", costs={"false_positive_cost": float("nan")}), "REQUEST_INVALID"),
        }
        for name, (request, code) in bad_requests.items():
            with self.subTest(name):
                with self.assertRaises(JobError) as caught:
                    coordinator.submit_job(dataset["dataset_id"], dataset["capability"], request)
                self.assertEqual((caught.exception.code, caught.exception.http_status), (code, 422))
        five = self.submit(coordinator, dataset, "sleep:0", models=list("abcde"), dataset_id=dataset["dataset_id"])
        self.assertEqual(self.wait_state(coordinator, five, "succeeded")["state"], "succeeded")
        self.assertEqual(len(self.status(coordinator, five)["models"]), 5)


class Ownership(CoordinatorCase):
    def test_wrong_foreign_missing_and_unknown_all_look_the_same(self):
        coordinator = self.make()
        mine, theirs = self.dataset(coordinator), self.dataset(coordinator)
        job = self.submit(coordinator, mine, "sleep:60")
        other_job = self.submit(coordinator, theirs, "sleep:60")
        wrong = ["x" * 43, other_job["capability"], theirs["capability"], mine["capability"][:-1], "", None, 5]
        keys = set()

        def collect(call):
            with self.assertRaises(JobError) as caught:
                call()
            keys.add(error_key(caught.exception))
            self.assertEqual(caught.exception.http_status, 404)

        for token in wrong[1:]:
            collect(lambda: self.submit_with(coordinator, mine["dataset_id"], token))
            collect(lambda: coordinator.delete_dataset(mine["dataset_id"], token))
        for token in [*wrong, mine["capability"]]:  # a dataset capability is not a job capability
            if token == job["capability"]:
                continue
            collect(lambda: coordinator.job_status(job["job_id"], token))
            collect(lambda: coordinator.cancel_job(job["job_id"], token))
            collect(lambda: coordinator.job_result(job["job_id"], token))
            collect(lambda: coordinator.job_export(job["job_id"], token, "report_json"))
        collect(lambda: coordinator.job_status("job_does_not_exist", job["capability"]))
        collect(lambda: coordinator.job_result("job_does_not_exist", job["capability"]))
        collect(lambda: coordinator.cancel_job("job_does_not_exist", job["capability"]))
        collect(lambda: self.submit_with(coordinator, "ds_does_not_exist", mine["capability"]))
        collect(lambda: coordinator.delete_dataset("ds_does_not_exist", mine["capability"]))
        collect(lambda: coordinator.job_status(None, job["capability"]))
        self.assertEqual(len(keys), 1, keys)
        (code, status, message, details), = keys
        self.assertEqual((code, status, details), ("NOT_FOUND", 404, "{}"))
        for forbidden in ("exist", "owner", "token", "capab", "expired", job["job_id"], mine["dataset_id"]):
            self.assertNotIn(forbidden, message.lower())
        # Nothing the wrong callers did had any effect.
        self.assertEqual(self.status(coordinator, job)["state"], "running")
        self.assertEqual(coordinator.stats()["datasets"], 2)
        coordinator.cancel_job(job["job_id"], job["capability"])
        coordinator.cancel_job(other_job["job_id"], other_job["capability"])

    def submit_with(self, coordinator, dataset_id, token):
        return coordinator.submit_job(dataset_id, token, self.request("sleep:1"))

    def test_an_unknown_export_kind_is_rejected_and_an_unfinished_job_has_no_export(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "sleep:60")
        with self.assertRaises(JobError) as caught:
            coordinator.job_export(job["job_id"], job["capability"], "../../etc/passwd")
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("EXPORT_UNKNOWN", 404))
        for kind in ("report_json", "report_html", "config_json"):
            with self.assertRaises(JobError) as caught:
                coordinator.job_export(job["job_id"], job["capability"], kind)
            self.assertEqual((caught.exception.code, caught.exception.http_status), ("RESULT_NOT_AVAILABLE", 409))
        state, report = coordinator.job_result(job["job_id"], job["capability"])
        self.assertIn(state, ("queued", "running"))
        self.assertIsNone(report)
        coordinator.cancel_job(job["job_id"], job["capability"])

    def test_deleting_a_dataset_cancels_its_job_and_removes_the_files(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        pids_path = self.pid_file()
        job = self.submit(coordinator, dataset, f"spawn:{pids_path}")
        pids = self.read_pids(pids_path)
        self.assertEqual(coordinator.delete_dataset(dataset["dataset_id"], dataset["capability"]), {"deleted": True})
        self.assertEqual(self.status(coordinator, job)["state"], "cancelled")
        self.assertDead(pids["runner"])
        self.assertDead(pids["grandchild"])
        self.assertEqual(coordinator.store.leftover_entries(), [])
        with self.assertRaises(JobError) as caught:
            self.submit(coordinator, dataset, "sleep:1")
        self.assertEqual(caught.exception.code, "NOT_FOUND")


class Cancellation(CoordinatorCase):
    def test_cancel_running_kills_the_whole_tree_and_cancel_again_changes_nothing(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        pids_path = self.pid_file()
        job = self.submit(coordinator, dataset, f"spawn:{pids_path}")
        pids = self.read_pids(pids_path)
        self.assertTrue(proc.pid_alive(pids["runner"]) and proc.pid_alive(pids["grandchild"]))
        self.assertEqual(self.status(coordinator, job)["state"], "running")
        started = time.monotonic()
        first = coordinator.cancel_job(job["job_id"], job["capability"])
        self.assertEqual(first["state"], "cancelled")
        self.assertLess(time.monotonic() - started, 15)
        self.assertFalse(proc.pid_alive(pids["runner"]))
        self.assertFalse(proc.pid_alive(pids["grandchild"]))
        for _ in range(3):
            again = coordinator.cancel_job(job["job_id"], job["capability"])
            self.assertEqual(again["state"], "cancelled")
        self.assertEqual(coordinator.job_result(job["job_id"], job["capability"]), ("cancelled", None))
        self.assertEqual(coordinator.stats()["heavy_running"], 0)
        self.assertEqual([name for name in coordinator.store.leftover_entries() if name.startswith("job_")], [])

    def test_a_straggler_left_behind_by_a_finished_job_is_killed_too(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        pids_path = self.pid_file()
        job = self.submit(coordinator, dataset, f"straggler:{pids_path}")
        pids = self.read_pids(pids_path)
        self.assertEqual(self.wait_state(coordinator, job, "succeeded")["state"], "succeeded")
        self.assertDead(pids["runner"])
        self.assertDead(pids["grandchild"], 20)

    def test_killing_a_tree_works_without_a_windows_job_object_too(self):
        base = Path(tempfile.mkdtemp(prefix="nojob-", dir=MODULE_BASE))
        self.addCleanup(store.remove_tree, base)
        job = base / "root" / "jobs" / "j1"
        job.mkdir(parents=True)
        pids_path = self.pid_file()
        (job / "request.json").write_text(json.dumps({"pid_file": str(pids_path), "seconds": 120}), encoding="utf-8")
        child = spawn_runner(JobConfig(allow_test_commands=True), "test_spawn", job, use_job=False)
        self.track(child.pid)
        pids = self.read_pids(pids_path)
        self.assertTrue(child.kill_tree(15))
        self.assertDead(pids["runner"])
        self.assertDead(pids["grandchild"])
        child.release()

    def test_cancel_queued_removes_it_without_touching_the_running_job(self):
        coordinator = self.make()
        first_dataset, second_dataset = self.dataset(coordinator), self.dataset(coordinator)
        running = self.submit(coordinator, first_dataset, "sleep:60")
        wait_for(lambda: self.status(coordinator, running)["state"] == "running", message="the first job to run")
        queued = self.submit(coordinator, second_dataset, "sleep:60")
        self.assertEqual(self.status(coordinator, queued)["state"], "queued")
        self.assertEqual(self.status(coordinator, queued)["queue_position"], 1)
        result = coordinator.cancel_job(queued["job_id"], queued["capability"])
        self.assertEqual((result["state"], result["queue_position"]), ("cancelled", None))
        self.assertEqual(self.status(coordinator, running)["state"], "running")
        self.assertEqual(coordinator.stats()["heavy_waiting"], 0)
        self.assertEqual(coordinator.cancel_job(queued["job_id"], queued["capability"])["state"], "cancelled")
        coordinator.cancel_job(running["job_id"], running["capability"])

    def test_cancel_after_the_job_finished_returns_the_final_state_unchanged(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "sleep:0")
        self.wait_state(coordinator, job, "succeeded")
        self.assertEqual(coordinator.cancel_job(job["job_id"], job["capability"])["state"], "succeeded")
        self.assertEqual(coordinator.job_result(job["job_id"], job["capability"])[0], "succeeded")

    def test_a_dataset_with_a_live_job_does_not_expire_while_it_runs(self):
        coordinator = self.make(dataset_idle_seconds=1.0)
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "sleep:4")
        time.sleep(2.5)
        self.assertEqual(coordinator.stats()["datasets"], 1)
        coordinator.cancel_job(job["job_id"], job["capability"])


class Timeouts(CoordinatorCase):
    def test_total_job_timeout_kills_the_tree_and_reports_timed_out(self):
        coordinator = self.make(job_timeout_seconds=3.0)
        dataset = self.dataset(coordinator)
        pids_path = self.pid_file()
        job = self.submit(coordinator, dataset, f"spawn:{pids_path}")
        pids = self.read_pids(pids_path)
        status = self.wait_state(coordinator, job, "timed_out", 30)
        self.assertEqual(status["reason"]["code"], "JOB_TIMEOUT")
        self.assertDead(pids["runner"])
        self.assertDead(pids["grandchild"])
        self.assertEqual(coordinator.job_result(job["job_id"], job["capability"]), ("timed_out", None))

    def test_a_legacy_job_honours_its_own_timeout(self):
        coordinator = self.make()
        handle = coordinator.run_heavy("test_sleep", {"seconds": 60}, timeout=2)
        with self.assertRaises(JobError) as caught:
            handle.wait(30)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("JOB_TIMEOUT", 503))
        self.assertEqual(handle.state, "timed_out")

    def test_light_task_timeout(self):
        coordinator = self.make(light_timeout_seconds=1.5)
        started = time.monotonic()
        with self.assertRaises(JobError) as caught:
            coordinator.run_light("test_sleep", {"seconds": 60})
        self.assertEqual(caught.exception.code, "JOB_TIMEOUT")
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(coordinator.stats()["light_running"], 0)

    def test_a_queued_job_that_waits_too_long_is_timed_out_with_queue_wait_exceeded(self):
        coordinator = self.make(queue_wait_seconds=1.5)
        first_dataset, second_dataset = self.dataset(coordinator), self.dataset(coordinator)
        running = self.submit(coordinator, first_dataset, "sleep:60")
        waiting = self.submit(coordinator, second_dataset, "sleep:60")
        status = self.wait_state(coordinator, waiting, "timed_out", 20)
        self.assertEqual(status["reason"]["code"], "QUEUE_WAIT_EXCEEDED")
        self.assertIsNone(status["queue_position"])
        self.assertEqual(self.status(coordinator, running)["state"], "running")
        coordinator.cancel_job(running["job_id"], running["capability"])
        self.assertEqual(coordinator.run_heavy("test_sleep", {"seconds": 0}).wait(60), {"slept": 0})  # slot is free


class Failures(CoordinatorCase):
    def test_a_hard_crash_is_failed_with_a_safe_code(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "crash:hard")
        status = self.wait_state(coordinator, job, "failed")
        self.assertEqual(status["reason"]["code"], "JOB_CRASHED")
        self.assertEqual(coordinator.job_result(job["job_id"], job["capability"]), ("failed", None))
        with self.assertRaises(JobError) as caught:
            coordinator.run_heavy("test_crash", {"mode": "hard_exit"}).wait(30)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("JOB_CRASHED", 500))

    def test_an_exception_in_the_child_never_leaks_a_path_a_traceback_or_a_token(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        secret = "tok_" + "s3cr3t" * 6
        job = self.submit(coordinator, dataset, f"crash:{secret}")
        status = self.wait_state(coordinator, job, "failed")
        body = json.dumps(status)
        self.assertEqual(status["reason"]["code"], "JOB_INTERNAL_ERROR")
        self.assertIn("RuntimeError", status["reason"]["message"])
        for forbidden in (secret, "secret", "Traceback", "place", "x.py", "C:\\", "boom", str(MODULE_BASE)):
            self.assertNotIn(forbidden, body)
        with self.assertRaises(JobError) as caught:
            coordinator.run_heavy("test_crash", {"mode": "exception", "leak_token": secret}).wait(30)
        text = json.dumps(caught.exception.to_dict())
        self.assertEqual(caught.exception.code, "JOB_INTERNAL_ERROR")
        for forbidden in (secret, "secret", "Traceback", "place", "x.py", "boom", str(MODULE_BASE)):
            self.assertNotIn(forbidden, text)
        with self.assertRaises(JobError) as caught:
            coordinator.run_heavy("test_crash", {"mode": "memory"}).wait(30)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("JOB_OUT_OF_MEMORY", 503))

    def test_a_reported_error_keeps_its_code(self):
        coordinator = self.make()
        with self.assertRaises(JobError) as caught:
            coordinator.run_heavy("test_crash", {"mode": "reported"}).wait(30)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("TEST_REPORTED", 422))

    def test_big_output_is_bounded_by_the_child_and_checked_again_by_the_coordinator(self):
        coordinator = self.make(max_result_bytes=1_000_000)
        for payload in ({"bytes": 3_000_000}, {"bytes": 3_000_000, "bypass": True}):
            with self.subTest(payload=payload):
                with self.assertRaises(JobError) as caught:
                    coordinator.run_heavy("test_big_output", payload).wait(60)
                self.assertEqual(caught.exception.code, "RESULT_TOO_LARGE")
                self.assertLess(len(json.dumps(caught.exception.to_dict())), 400)
        small = coordinator.run_heavy("test_big_output", {"bytes": 200_000}).wait(60)
        self.assertEqual(len(small["blob"]), 200_000)
        self.assertLess(coordinator.stats()["stored_bytes"], 10_000)  # nothing large stays on disk

    def test_a_job_that_is_not_collected_is_dropped_by_the_sweep(self):
        coordinator = self.make()
        handle = coordinator.run_heavy("test_sleep", {"seconds": 0})
        wait_for(lambda: handle.state == "succeeded", message="the job to finish")
        self.assertEqual(coordinator.stats()["jobs"], 1)
        handle.wait(5)
        self.assertEqual(coordinator.stats()["jobs"], 0)


class ModelStatusMapping(CoordinatorCase):
    CASES = [
        ("a=success,b=success", "succeeded", None),
        ("a=success,b=timeout", "partial", "SOME_MODELS_FAILED"),
        ("a=failed,b=success", "partial", "SOME_MODELS_FAILED"),
        ("a=success,b=failed,c=timeout", "partial", "SOME_MODELS_FAILED"),
        ("a=failed,b=failed", "failed", "MODEL_FIT_ERROR"),
        ("a=failed,b=timeout", "failed", "ALL_MODELS_FAILED"),
        ("a=timeout,b=timeout", "timed_out", "MODEL_TIMEOUT"),
    ]

    def test_training_model_statuses_map_to_succeeded_partial_failed_and_timed_out(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        for spec, state, code in self.CASES:
            with self.subTest(spec):
                models = [item.split("=")[0] for item in spec.split(",")]
                job = self.submit(coordinator, dataset, f"statuses:{spec}", models=models)
                status = self.wait_state(coordinator, job, tuple(states.FINISHED_STATES))
                self.assertEqual(status["state"], state)
                self.assertEqual(None if status["reason"] is None else status["reason"]["code"], code)
                expected = {name: {"success": "succeeded", "timeout": "timed_out", "failed": "failed"}[value]
                            for name, value in (item.split("=") for item in spec.split(","))}
                self.assertEqual({item["model"]: item["status"] for item in status["models"]}, expected)
                for item in status["models"]:
                    self.assertEqual("reason_code" in item, item["status"] != "succeeded")
                result_state, report = coordinator.job_result(job["job_id"], job["capability"])
                self.assertEqual(result_state, state)
                if state in ("succeeded", "partial"):
                    self.assertEqual(sorted(report["outcomes"]), sorted(models))  # failed models are never dropped
                    self.assertEqual(report["schema_version"], "training-1.1")
                else:
                    self.assertIsNone(report)

    def test_partial_and_succeeded_jobs_export_the_report_and_the_settings(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "statuses:a=success,b=failed", models=["a", "b"], seed=11)
        self.wait_state(coordinator, job, "partial")
        data, content_type, filename = coordinator.job_export(job["job_id"], job["capability"], "report_json")
        self.assertEqual((content_type, filename), ("application/json", "fyd-report.json"))
        self.assertEqual(sorted(json.loads(data)["outcomes"]), ["a", "b"])
        data, content_type, filename = coordinator.job_export(job["job_id"], job["capability"], "config_json")
        config = json.loads(data)
        self.assertEqual((content_type, filename), ("application/json", "fyd-config.json"))
        self.assertEqual((config["request"]["seed"], config["request"]["models"]), (11, ["a", "b"]))
        self.assertEqual(config["report_schema_version"], "training-1.1")
        for forbidden in (job["capability"], dataset["capability"], str(MODULE_BASE)):
            self.assertNotIn(forbidden, data.decode("utf-8"))


class Progress(CoordinatorCase):
    def test_progress_follows_reported_events_only_and_never_holds_a_percentage(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "steps:0.45", models=["a", "b", "c"])
        seen, bodies = [], []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = self.status(coordinator, job)
            bodies.append(json.dumps(status))
            if status["progress"]:
                seen.append(status)
            if status["state"] not in ("queued", "running"):
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "succeeded")
        stages = []
        for item in seen:
            if not stages or stages[-1] != item["progress"]["stage"]:
                stages.append(item["progress"]["stage"])
        for needed in ("splitting", "fitting", "reporting", "done"):
            self.assertIn(needed, stages)
        self.assertLess(stages.index("splitting"), stages.index("fitting"))
        self.assertLess(stages.index("fitting"), stages.index("reporting"))
        self.assertLess(stages.index("reporting"), stages.index("done"))
        counts = [len(item["progress"]["models_completed"]) for item in seen]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 3)
        self.assertIn({"a"}, [{item["progress"]["current_model"]} for item in seen if item["progress"]["current_model"]])
        self.assertTrue(any(item["progress"]["current_model"] is None and item["progress"]["stage"] == "fitting"
                            for item in seen))
        self.assertEqual(seen[-1]["progress"]["models_selected"], ["a", "b", "c"])
        elapsed = [item["elapsed_seconds"] for item in seen]
        self.assertEqual(elapsed, sorted(elapsed))
        self.assertEqual(set(seen[0]["progress"]), {"stage", "models_selected", "models_completed", "current_model",
                                                    "elapsed_seconds", "limits", "warnings"})
        for body in bodies:
            self.assertNotRegex(body.lower(), r"percent|fraction|\beta\b|remaining")

    def test_without_events_the_stage_does_not_advance_however_long_it_takes(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "sleep:3.5")
        wait_for(lambda: (self.status(coordinator, job)["progress"] or {}).get("stage") == "parsing", message="a stage")
        time.sleep(2.0)
        status = self.status(coordinator, job)
        self.assertGreaterEqual(status["elapsed_seconds"], 1.9)
        self.assertEqual(status["progress"]["stage"], "parsing")
        self.assertEqual(status["progress"]["models_completed"], [])
        self.assertNotRegex(json.dumps(status).lower(), r"percent|fraction")
        coordinator.cancel_job(job["job_id"], job["capability"])

    def test_the_progress_file_is_untrusted_and_only_documented_keys_pass(self):
        sanitized = Coordinator._sanitize_progress({
            "stage": "fitting", "models_selected": ["a", {"x": 1}, 3], "current_model": "a" * 500,
            "models_completed": [{"model": "a", "status": "succeeded", "extra": "C:\\Users\\x"}, "junk"],
            "warnings": ["see C:\\Users\\someone\\data.csv now", 5], "limits": {"model_timeout_seconds": 1, "evil": 2},
            "percent_complete": 55, "elapsed_seconds": -1})
        self.assertEqual(set(sanitized), {"stage", "models_selected", "models_completed", "current_model", "limits", "warnings"})
        self.assertEqual(sanitized["models_completed"], [{"model": "a", "status": "succeeded"}])
        self.assertLessEqual(len(sanitized["current_model"]), 64)
        self.assertEqual(sanitized["limits"], {"model_timeout_seconds": 1})
        self.assertNotIn("Users", json.dumps(sanitized))
        self.assertIsNone(Coordinator._sanitize_progress({"stage": "bogus"}))
        self.assertIsNone(Coordinator._sanitize_progress([1]))


class Lifetimes(CoordinatorCase):
    def test_finished_jobs_expire_leave_a_tombstone_and_the_tombstone_ends(self):
        coordinator = self.make(dataset_idle_seconds=3.0, finished_job_seconds=1.0, tombstone_seconds=4.0)
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "sleep:0")
        status = self.wait_state(coordinator, job, "succeeded")
        self.assertIsNotNone(status["expires_at"])
        job_dir = coordinator.store.job_dir(job["job_id"])
        self.assertTrue((job_dir / "result.json").exists())
        status = self.wait_state(coordinator, job, "expired", 20)
        self.assertFalse(job_dir.exists())
        self.assertNotIn(job["job_id"], coordinator.store.leftover_entries())
        self.assertEqual(coordinator.job_result(job["job_id"], job["capability"]), ("expired", None))
        with self.assertRaises(JobError) as caught:
            coordinator.job_export(job["job_id"], job["capability"], "report_json")
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("JOB_EXPIRED", 410))
        self.assertEqual(coordinator.cancel_job(job["job_id"], job["capability"])["state"], "expired")
        with self.assertRaises(JobError) as caught:  # a wrong token still learns nothing
            coordinator.job_status(job["job_id"], "x" * 43)
        self.assertEqual(caught.exception.code, "NOT_FOUND")
        self.assertGreaterEqual(coordinator.stats()["tombstones"], 1)
        wait_for(lambda: coordinator.stats()["datasets"] == 0, 20, message="the idle dataset to expire")
        self.assertNotIn(dataset["dataset_id"], coordinator.store.leftover_entries())
        with self.assertRaises(JobError) as caught:
            self.submit(coordinator, dataset, "sleep:1")
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("DATASET_EXPIRED", 410))
        with self.assertRaises(JobError) as caught:
            coordinator.submit_job(dataset["dataset_id"], "x" * 43, self.request("sleep:1"))
        self.assertEqual(caught.exception.code, "NOT_FOUND")
        wait_for(lambda: coordinator.stats()["tombstones"] == 0, 30, message="the tombstones to end")
        with self.assertRaises(JobError) as caught:
            coordinator.job_status(job["job_id"], job["capability"])
        self.assertEqual(caught.exception.code, "NOT_FOUND")

    def test_the_storage_cap_rejects_new_uploads_with_503(self):
        coordinator = self.make(max_stored_bytes=2_500_000, max_result_bytes=1_000_000)
        body = b"a,b\n" + b"1,2\n" * 250_000  # about 1 MB
        first = coordinator.create_dataset(body, "one.csv")
        with self.assertRaises(JobError) as caught:
            coordinator.create_dataset(body, "two.csv")
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("STORAGE_FULL", 503))
        self.assertEqual(coordinator.stats()["datasets"], 1)
        self.assertEqual(len(coordinator.store.leftover_entries()), 1)
        coordinator.delete_dataset(first["dataset_id"], first["capability"])
        coordinator.create_dataset(body, "three.csv")

    def test_the_sweep_enforces_the_byte_cap_by_expiring_the_oldest_finished_job(self):
        coordinator = self.make(max_stored_bytes=2_500_000, max_result_bytes=1_000_000, finished_job_seconds=3600)
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "sleep:0")
        self.wait_state(coordinator, job, "succeeded")
        with self.assertLogs("fyd.jobs", level="WARNING") as logged:
            (coordinator.store.job_dir(job["job_id"]) / "grown.bin").write_bytes(b"x" * 3_000_000)
            self.wait_state(coordinator, job, "expired", 20)
        self.assertTrue(any("storage cap reached" in line for line in logged.output))
        self.assertFalse(coordinator.store.job_dir(job["job_id"]).exists())
        self.assertLess(coordinator.stats()["stored_bytes"], 2_500_000)


class ThreadSafety(CoordinatorCase):
    def test_many_threads_polling_and_cancelling_never_break_the_state(self):
        coordinator = self.make()
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "sleep:60")
        observed, errors = [], []

        def poll():
            for _ in range(25):
                try:
                    observed.append(self.status(coordinator, job)["state"])
                    coordinator.stats()
                except Exception as exc:  # pragma: no cover - only on failure
                    errors.append(exc)
                time.sleep(0.02)

        def cancel():
            try:
                coordinator.cancel_job(job["job_id"], job["capability"])
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=poll) for _ in range(6)] + [threading.Thread(target=cancel) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(self.status(coordinator, job)["state"], "cancelled")
        for first, second in zip(observed, observed[1:]):
            self.assertTrue(first == second or states.can_transition(first, second) or first == "queued", (first, second))


class Secrets(CoordinatorCase):
    def test_no_capability_filename_or_secret_looking_value_reaches_a_log_record_or_an_error_body(self):
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(self.format(record))

        handler = Capture(level=logging.DEBUG)
        logger = logging.getLogger("fyd.jobs")
        previous = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(logger.setLevel, previous)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        self.addCleanup(root_logger.removeHandler, handler)

        coordinator = self.make()
        dataset = coordinator.create_dataset(b"a,b\n1,2\n", "patients-secret-name.csv")
        job = self.submit(coordinator, dataset, "crash:hard")
        self.wait_state(coordinator, job, "failed")
        bodies = []
        for call in (lambda: coordinator.job_status(job["job_id"], "x" * 43),
                     lambda: coordinator.submit_job(dataset["dataset_id"], None, self.request("t")),
                     lambda: coordinator.job_export(job["job_id"], job["capability"], "report_json"),
                     lambda: coordinator.create_dataset(b"x" * 20_000_000, "huge-secret.csv")):
            with self.assertRaises(JobError) as caught:
                call()
            bodies.append(json.dumps(caught.exception.to_dict()))
        coordinator.delete_dataset(dataset["dataset_id"], dataset["capability"])
        coordinator.stop(10)
        secrets_seen = [dataset["capability"], job["capability"], tokens.digest_of(dataset["capability"]),
                        tokens.digest_of(job["capability"]), "patients-secret-name", "huge-secret"]
        self.assertGreater(len(records), 5)
        text = "\n".join(records + bodies)
        for value in secrets_seen:
            self.assertNotIn(value, text)
        self.assertNotRegex(text, r"[A-Za-z0-9_-]{40,}")
        self.assertNotIn(str(MODULE_BASE), text)


# ============================================================================== parent death

class ParentDeath(CoordinatorCase):
    def run_helper(self, use_job: bool):
        base = Path(tempfile.mkdtemp(prefix="parent-", dir=MODULE_BASE))
        self.addCleanup(store.remove_tree, base)
        helper = subprocess.Popen([sys.executable, "-B", str(PARENT_HELPER), str(base), "1" if use_job else "0"],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def finish_helper():
            if helper.poll() is None:
                helper.kill()
            helper.wait()
        self.addCleanup(finish_helper)
        wait_for(lambda: (base / "helper.pid").exists() and (base / "pids.json").exists(), 60, message="the helper")
        helper_pid = int((base / "helper.pid").read_text(encoding="utf-8"))
        self.track(helper_pid)
        pids = self.read_pids(base / "pids.json")
        self.assertTrue(proc.pid_alive(pids["runner"]) and proc.pid_alive(pids["grandchild"]))
        kill_only(helper_pid)
        self.assertDead(helper_pid)
        return pids

    def test_the_watchdog_ends_the_child_and_its_grandchild_when_the_parent_dies(self):
        pids = self.run_helper(use_job=False)  # no Windows job object: only the watchdog can do this
        self.assertDead(pids["runner"], 30)
        self.assertDead(pids["grandchild"], 30)

    @unittest.skipUnless(os.name == "nt", "Windows job objects only")
    def test_on_windows_the_job_object_also_ends_the_tree_when_the_parent_dies(self):
        pids = self.run_helper(use_job=True)
        self.assertDead(pids["runner"], 30)
        self.assertDead(pids["grandchild"], 30)

    def test_pid_alive_reports_a_running_and_an_exited_process(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", MARKER],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.track(child.pid)
        self.assertTrue(proc.pid_alive(child.pid))
        child.kill()
        child.wait()
        self.assertFalse(proc.pid_alive(child.pid))
        self.assertFalse(proc.pid_alive(0))
        self.assertFalse(proc.pid_alive(-5))


# ============================================================================== real work

def synthetic_csv(rows: int = 300, positives: int = 40, categorical_only: bool = False) -> bytes:
    columns = mixed_table(n=rows, positives=positives)
    if categorical_only:
        for name in ("age", "income"):
            del columns[name]
    return table_to_csv(columns)


class RealWork(CoordinatorCase):
    """Real runner, real subprocesses, real profile and real training on a synthetic table of 300 rows."""

    TRAIN = {"target": "label", "positive_label": "fraud", "negative_label": "ok", "seed": 7}

    @classmethod
    def setUpClass(cls):
        cls.base = Path(tempfile.mkdtemp(prefix="real-", dir=MODULE_BASE))
        cls.coordinator = Coordinator(JobConfig(temp_root=str(cls.base / "root"), sweep_interval_seconds=0.5))
        cls.coordinator.start()
        cls.dataset = cls.coordinator.create_dataset(synthetic_csv(), "synthetic.csv")

    @classmethod
    def tearDownClass(cls):
        cls.coordinator.stop(10)
        if cls.coordinator.store.root.exists():
            LEAKED_ROOTS.append(cls.coordinator.store.root)
        store.remove_tree(cls.base)

    def run_job(self, request: dict, dataset: dict | None = None, wait=("succeeded", "partial", "failed", "timed_out")):
        dataset = dataset or self.dataset
        job = self.coordinator.submit_job(dataset["dataset_id"], dataset["capability"], request)
        status = self.wait_state(self.coordinator, job, wait, 180)
        return job, status

    def status(self, coordinator, job) -> dict:
        return self.coordinator.job_status(job["job_id"], job["capability"])

    def test_the_real_profile_of_a_synthetic_mixed_table(self):
        dataset = self.dataset
        self.assertEqual(set(dataset), {"dataset_id", "capability", "profile", "expires_at", "limits"})
        profile = dataset["profile"]
        self.assertEqual((profile["schema"], profile["rows"], profile["columns"]), ("app-profile-1.0", 300, 5))
        self.assertEqual(profile["target"]["suggestion"]["column"], "label")
        self.assertEqual(profile["classes"]["suggested_positive"]["label"], "fraud")
        self.assertEqual(profile["classes"]["suggested_positive"]["status"], "suggested, needs confirmation")
        self.assertEqual(profile["limits"]["status"], "ok")
        self.assertEqual(profile["limits"]["configured"]["max_rows"], 10000)
        self.assertEqual(dataset["limits"]["max_models_per_job"], 5)
        self.assertRegex(dataset["expires_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertNotIn(str(MODULE_BASE), json.dumps(dataset))
        self.assertNotIn(dataset["capability"], json.dumps(profile))

    def test_an_upload_that_cannot_be_read_is_refused_with_a_typed_error_and_leaves_nothing(self):
        before = self.coordinator.stats()["datasets"]
        entries = sorted(self.coordinator.store.leftover_entries())
        for raw, name in ((b"a,b\n", "header-only.csv"), (b"not a workbook", "broken.xlsx"), (b"{bad", "broken.json")):
            with self.assertRaises(JobError) as caught:
                self.coordinator.create_dataset(raw, name)
            self.assertEqual(caught.exception.http_status, 422, caught.exception.to_dict())
            self.assertEqual(caught.exception.code, "APP_INPUT_UNREADABLE")
            self.assertNotIn(str(MODULE_BASE), json.dumps(caught.exception.to_dict()))
            self.assertNotIn("Traceback", json.dumps(caught.exception.to_dict()))
        self.assertEqual(self.coordinator.stats()["datasets"], before)
        self.assertEqual(sorted(self.coordinator.store.leftover_entries()), entries)

    def test_a_tiny_training_job_end_to_end_with_every_export(self):
        job, status = self.run_job({**self.TRAIN, "models": ["logistic", "rf_weighted"]})
        self.assertEqual(status["state"], "succeeded", status)
        self.assertEqual([item["status"] for item in status["models"]], ["succeeded", "succeeded"])
        self.assertEqual(status["progress"]["stage"], "done")
        self.assertEqual([item["model"] for item in status["progress"]["models_completed"]], ["logistic", "rf_weighted"])
        state, report = self.coordinator.job_result(job["job_id"], job["capability"])
        self.assertEqual(state, "succeeded")
        self.assertEqual(report["schema_version"], "training-1.1")
        self.assertEqual({name: row["status"] for name, row in report["outcomes"].items()},
                         {"logistic": "success", "rf_weighted": "success"})
        self.assertEqual(report["input_parsing"]["rows_used"], 300)
        self.assertEqual(report["label_mapping"]["positive_label"], "fraud")
        self.assertTrue(all("diagnostic" not in row for row in report["outcomes"].values()))
        encoded = json.dumps(report)
        for forbidden in (str(MODULE_BASE), tempfile.gettempdir(), job["capability"], self.dataset["capability"]):
            self.assertNotIn(forbidden, encoded)
        data, content_type, filename = self.coordinator.job_export(job["job_id"], job["capability"], "report_json")
        self.assertEqual(json.loads(data)["schema_version"], "training-1.1")
        html, content_type, filename = self.coordinator.job_export(job["job_id"], job["capability"], "report_html")
        self.assertEqual((content_type, filename), ("text/html; charset=utf-8", "fyd-report.html"))
        self.assertIn(b"<html", html.lower())
        self.assertNotIn(job["capability"].encode(), html)
        data, _, _ = self.coordinator.job_export(job["job_id"], job["capability"], "config_json")
        config = json.loads(data)
        self.assertEqual(config["request"]["models"], ["logistic", "rf_weighted"])
        self.assertEqual(config["input_sha256"], report["input_sha256"])
        self.assertEqual(self.coordinator.stats()["heavy_running"], 0)

    def test_a_real_partial_run_keeps_the_failed_model_in_the_report_with_its_reason(self):
        dataset = self.coordinator.create_dataset(synthetic_csv(categorical_only=True), "categorical.csv")
        self.addCleanup(self.coordinator.delete_dataset, dataset["dataset_id"], dataset["capability"])
        job, status = self.run_job({**self.TRAIN, "models": ["logistic", "smote_rf"]}, dataset)
        self.assertEqual(status["state"], "partial", status)
        self.assertEqual(status["reason"]["code"], "SOME_MODELS_FAILED")
        self.assertEqual({item["model"]: item["status"] for item in status["models"]},
                         {"logistic": "succeeded", "smote_rf": "failed"})
        self.assertEqual([item["reason_code"] for item in status["models"] if item["status"] == "failed"],
                         ["RESAMPLING_NO_NUMERIC_FEATURES"])
        state, report = self.coordinator.job_result(job["job_id"], job["capability"])
        self.assertEqual(state, "partial")
        self.assertEqual(report["outcomes"]["smote_rf"]["status"], "failed")
        self.assertEqual(report["outcomes"]["smote_rf"]["reason_code"], "RESAMPLING_NO_NUMERIC_FEATURES")
        self.assertEqual({item["model"] for item in report["failures"]}, {"smote_rf"})
        self.assertTrue(all("diagnostic" not in row for row in report["outcomes"].values()))
        self.assertNotIn(str(MODULE_BASE), json.dumps(report))
        html, _, _ = self.coordinator.job_export(job["job_id"], job["capability"], "report_html")
        self.assertIn(b"smote_rf", html)

    def test_a_request_the_package_refuses_ends_failed_with_the_package_code_and_a_safe_message(self):
        job, status = self.run_job({**self.TRAIN, "target": "no_such_column", "models": ["logistic"]})
        self.assertEqual(status["state"], "failed")
        self.assertRegex(status["reason"]["code"], r"^[A-Z][A-Z0-9_]+$")
        self.assertNotEqual(status["reason"]["code"], "JOB_INTERNAL_ERROR")
        self.assertNotIn(str(MODULE_BASE), json.dumps(status))
        self.assertNotIn("Traceback", json.dumps(status))
        job, status = self.run_job({**self.TRAIN, "models": ["no_such_model"]})
        self.assertEqual((status["state"], status["reason"]["code"]), ("failed", "OPTION_INVALID"))
        job, status = self.run_job({**self.TRAIN, "positive_label": "never_seen", "models": ["logistic"]})
        self.assertEqual(status["state"], "failed")
        self.assertEqual(self.coordinator.job_result(job["job_id"], job["capability"]), ("failed", None))

    def test_the_training_job_can_be_cancelled_while_a_real_model_is_fitting(self):
        job = self.coordinator.submit_job(self.dataset["dataset_id"], self.dataset["capability"],
                                          {**self.TRAIN, "models": ["erf", "csrboost", "balanced_rf"]})
        wait_for(lambda: self.status(None, job)["state"] == "running", 60, message="the job to start")
        wait_for(lambda: (self.status(None, job)["progress"] or {}).get("current_model"), 120, message="a model to fit")
        started = time.monotonic()
        status = self.coordinator.cancel_job(job["job_id"], job["capability"])
        self.assertEqual(status["state"], "cancelled")
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(self.coordinator.job_result(job["job_id"], job["capability"]), ("cancelled", None))
        worker_dirs = [path for path in self.coordinator.store.root.rglob("rare-event-worker-*")]
        self.assertEqual(worker_dirs, [])  # the package's worker directory went with the job


class RealModelTimeout(CoordinatorCase):
    def test_a_model_that_exceeds_the_per_model_limit_ends_the_job_timed_out(self):
        coordinator = self.make(fake=False, allow_test_commands=False, model_timeout_seconds=1)
        dataset = coordinator.create_dataset(synthetic_csv(rows=1500, positives=150), "bigger.csv")
        job = coordinator.submit_job(dataset["dataset_id"], dataset["capability"],
                                     {"target": "label", "positive_label": "fraud", "negative_label": "ok",
                                      "models": ["erf"]})
        status = self.wait_state(coordinator, job, tuple(states.FINISHED_STATES), 120)
        self.assertEqual(status["state"], "timed_out", status)
        self.assertEqual(status["reason"]["code"], "MODEL_TIMEOUT")
        self.assertEqual(status["models"], [{"model": "erf", "status": "timed_out", "reason_code": "MODEL_TIMEOUT"}])
        self.assertEqual(coordinator.job_result(job["job_id"], job["capability"]), ("timed_out", None))
        self.assertEqual(status["limits"]["per_model_timeout_seconds"], 1)


@unittest.skipIf(legacy_engine is None, LEGACY_REASON or "")
class RealLegacy(CoordinatorCase):
    CSV = "\n".join(["age,income,color,label"] + [
        f"{20 + (i * 7) % 40},{1000 + (i * 37) % 500},{('red', 'green', 'blue')[i % 3]},{'rare' if i % 9 == 0 else 'common'}"
        for i in range(180)]) + "\n"

    @classmethod
    def setUpClass(cls):
        cls.base = Path(tempfile.mkdtemp(prefix="legacy-", dir=MODULE_BASE))
        cls.coordinator = Coordinator(JobConfig(temp_root=str(cls.base / "root")))
        cls.coordinator.start()

    @classmethod
    def tearDownClass(cls):
        cls.coordinator.stop(10)
        if cls.coordinator.store.root.exists():
            LEAKED_ROOTS.append(cls.coordinator.store.root)
        store.remove_tree(cls.base)

    def test_legacy_profile_and_convert_match_the_engine_called_directly(self):
        import base64
        expected = json.loads(json.dumps(legacy_engine.profile(legacy_engine.read_csv(self.CSV))))
        self.assertEqual(self.coordinator.run_light("legacy_profile", {"csv": self.CSV}), expected)
        raw = self.CSV.encode("utf-8")
        text, notes = legacy_engine.any_to_csv(raw, "x.csv")
        converted = self.coordinator.run_light("legacy_convert", {"b64": base64.b64encode(raw).decode(), "name": "x.csv"})
        self.assertEqual(converted, {"ok": True, "csv": text, "notes": notes})

    @staticmethod
    def engine_error(raw: bytes, name: str) -> Exception:
        try:
            legacy_engine.any_to_csv(raw, name)
        except Exception as exc:  # noqa: BLE001 - the same call the live server makes
            return exc
        raise AssertionError("the engine accepted the input")

    def test_legacy_errors_keep_the_messages_the_live_server_sends(self):
        import base64
        with self.assertRaises(JobError) as caught:
            self.coordinator.run_light("legacy_profile", {})
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("LEGACY_BAD_REQUEST", 400))
        self.assertEqual(caught.exception.message, "The request is missing a required field, or names a column that "
                                                   "is not in the file: 'csv'.")
        with self.assertRaises(JobError) as caught:
            self.coordinator.run_heavy("legacy_analyze", {"csv": self.CSV}).wait(60)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("LEGACY_ERROR", 400))
        self.assertEqual(caught.exception.message, "Could not read that file well enough to find a target column. "
                                                   "Check the file's format.")
        with self.assertRaises(JobError) as caught:
            self.coordinator.run_light("legacy_convert", {"b64": base64.b64encode(b"not a workbook").decode(),
                                                          "name": "x.xlsx"})
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("LEGACY_ERROR", 400))
        self.assertEqual(caught.exception.message, str(self.engine_error(b"not a workbook", "x.xlsx")))
        self.assertNotIn(str(MODULE_BASE), caught.exception.message)

    def test_legacy_analyze_runs_in_the_heavy_lane_and_returns_the_engine_result(self):
        handle = self.coordinator.run_heavy("legacy_analyze", {"csv": self.CSV, "target": "label", "method": "smote",
                                                               "do_classify": False, "folds": 1})
        self.assertIsInstance(handle, HeavyHandle)
        result = handle.wait(120)
        self.assertIsInstance(result, dict)
        self.assertTrue(result)
        self.assertEqual(handle.state, "succeeded")
        scored = self.coordinator.run_heavy("legacy_score", {"train_csv": self.CSV, "target": "label",
                                                             "method": "smote", "classifier": "logreg",
                                                             "score_csv": self.CSV}).wait(120)
        self.assertIsInstance(scored, dict)


# ============================================================================== F1: the legacy result cap

class LegacyResultCap(CoordinatorCase):
    """Legacy commands have their own, larger result cap; the v1 commands keep the 8 MB one."""

    def test_a_legacy_result_between_8_and_32_mb_is_relayed_in_both_lanes(self):
        coordinator = self.make()
        size = 12_000_000    # above the 8 MB v1 cap, below the 32 MB legacy cap
        heavy = coordinator.run_heavy("test_legacy_big_output", {"bytes": size}).wait(120)
        self.assertEqual((heavy["ok"], len(heavy["blob"])), (True, size))
        light = coordinator.run_light("test_legacy_big_output", {"bytes": size})
        self.assertEqual((light["ok"], len(light["blob"])), (True, size))
        self.assertLess(coordinator.stats()["stored_bytes"], 10_000)   # the large file is gone once relayed

    def test_the_same_size_from_a_v1_command_is_still_refused(self):
        coordinator = self.make()
        with self.assertRaises(JobError) as caught:
            coordinator.run_heavy("test_big_output", {"bytes": 12_000_000}).wait(120)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("RESULT_TOO_LARGE", 422))
        for command in ("legacy_profile", "legacy_analyze", "legacy_score", "legacy_convert"):
            self.assertEqual(coordinator.result_cap(command), 32 * 1024 * 1024, command)
        for command in ("profile", "train", "test_big_output"):
            self.assertEqual(coordinator.result_cap(command), 8 * 1024 * 1024, command)

    def test_a_legacy_result_over_32_mb_fails_safely_with_a_typed_error_in_the_legacy_shape(self):
        coordinator = self.make()
        for payload in ({"bytes": 34_000_000}, {"bytes": 34_000_000, "bypass": True}):   # child check, parent check
            with self.subTest(payload=payload):
                with self.assertRaises(JobError) as caught:
                    coordinator.run_heavy("test_legacy_big_output", payload).wait(120)
                self.assertEqual((caught.exception.code, caught.exception.http_status), ("RESULT_TOO_LARGE", 400))
                self.assertIn("32 MB", caught.exception.message)
                self.assertLess(len(json.dumps(caught.exception.to_dict())), 400)
                self.assertEqual(coordinator.stats()["heavy_running"], 0)
        with self.assertRaises(JobError) as caught:   # the light lane answers the same way
            coordinator.run_light("test_legacy_big_output", {"bytes": 34_000_000})
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("RESULT_TOO_LARGE", 400))
        self.assertLess(coordinator.stats()["stored_bytes"], 10_000)   # nothing large stays on disk
        self.assertEqual(coordinator.run_heavy("test_sleep", {"seconds": 0}).wait(60), {"slept": 0})   # still usable

    def test_a_smaller_configured_cap_applies_and_says_so(self):
        coordinator = self.make(max_legacy_result_bytes=500_000)
        with self.assertRaises(JobError) as caught:
            coordinator.run_heavy("test_legacy_big_output", {"bytes": 600_000}).wait(60)
        self.assertEqual(caught.exception.code, "RESULT_TOO_LARGE")
        self.assertIn("488 KB", caught.exception.message)
        self.assertEqual(len(coordinator.run_heavy("test_legacy_big_output", {"bytes": 300_000}).wait(60)["blob"]), 300_000)

    def test_a_legacy_job_reserves_its_own_cap_and_the_storage_cap_still_holds(self):
        coordinator = self.make(max_stored_bytes=10_000_000, max_legacy_result_bytes=6_000_000,
                                max_result_bytes=1_000_000)
        self.assertEqual(coordinator._reserve_for("legacy_analyze"), 6_000_000)
        self.assertEqual(coordinator._reserve_for("legacy_convert"), 6_000_000)
        self.assertEqual(coordinator._reserve_for("legacy_profile"), 4 * 1024 * 1024)   # statistics only
        self.assertEqual(coordinator._reserve_for("train"), 0)
        first = coordinator.run_heavy("test_legacy_big_output", {"bytes": 1000, "delay": 3})
        wait_for(lambda: coordinator.stats()["heavy_running"] == 1, message="the first legacy job to run")
        with self.assertRaises(JobError) as caught:   # 6 MB held + 6 MB wanted is over 10 MB
            coordinator.run_light("test_legacy_big_output", {"bytes": 1000})
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("STORAGE_FULL", 503))
        with self.assertRaises(JobError) as caught:   # a v1 upload must not take the space the legacy job holds
            coordinator.create_dataset(b"a,b\n" + b"1,2\n" * 1_000_000, "big.csv")
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("STORAGE_FULL", 503))
        self.assertEqual(first.wait(60)["ok"], True)
        self.assertEqual(coordinator.stats()["jobs"], 0)
        again = coordinator.run_light("test_legacy_big_output", {"bytes": 1000})   # the reserve was released
        self.assertEqual(again["ok"], True)
        coordinator.create_dataset(b"a,b\n" + b"1,2\n" * 1_000_000, "big.csv")

    def test_the_default_64_mb_storage_cap_leaves_room_for_a_large_legacy_result(self):
        config = JobConfig()
        self.assertGreaterEqual(config.max_stored_bytes, config.max_legacy_result_bytes + config.max_result_bytes)
        coordinator = self.make()
        handle = coordinator.run_heavy("test_legacy_big_output", {"bytes": 20_000_000, "delay": 0.2})
        self.assertEqual(len(handle.wait(120)["blob"]), 20_000_000)

    def test_a_legacy_refusal_is_400_and_a_v1_refusal_keeps_its_status(self):
        too_large = {"code": "RESULT_TOO_LARGE", "message": "m"}
        memory = {"code": "MEMORY_LIMIT_EXCEEDED", "message": "m"}
        for command in ("legacy_analyze", "legacy_score", "legacy_profile", "legacy_convert"):
            for reason in (too_large, memory):
                self.assertEqual(Coordinator._error_for("failed", reason, command).http_status, 400, (command, reason))
        for command in ("profile", "train", "test_memory"):
            for reason in (too_large, memory):
                self.assertEqual(Coordinator._error_for("failed", reason, command).http_status, 422, (command, reason))


# ============================================================================== F2: the child memory ceiling

def stat_line(pid: int, comm: str, ppid: int, pgrp: int, rss_pages: int, state: str = "S") -> str:
    """A ``/proc/<pid>/stat`` line (52 fields, as on a current kernel); the parser reads state, ppid, pgrp and rss."""
    fields = [state, ppid, pgrp, pgrp, 0, -1, 4194560, 100, 0, 0, 0, 5, 1, 0, 0, 20, 0, 3, 0, 98765, 123456789, rss_pages]
    fields += [18446744073709551615, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0]
    return f"{pid} ({comm}) " + " ".join(str(item) for item in fields) + "\n"


class ProcMemoryParsing(unittest.TestCase):
    """The POSIX sampling code, driven with fixture text so it can be checked on any platform."""

    def test_parse_proc_stat_reads_ids_and_rss_even_with_odd_command_names(self):
        self.assertEqual(proc.parse_proc_stat(stat_line(4242, "python3", 1, 4242, 25_000)), (4242, 1, 4242, 25_000))
        for comm in ("a b c", "x) S 1 1 1 (y", "weird)(name", "((("):
            self.assertEqual(proc.parse_proc_stat(stat_line(77, comm, 5, 5, 123)), (77, 5, 5, 123), comm)

    def test_parse_proc_stat_rejects_text_that_is_not_a_stat_line(self):
        for text in ("", "garbage", "12 (short) S 1", "x (name) S a b", "12 (name) S 1 2 3 " + "4 " * 5):
            self.assertIsNone(proc.parse_proc_stat(text), repr(text))

    def test_tree_rss_sums_the_root_its_descendants_and_its_process_group_only(self):
        rows = [proc.parse_proc_stat(line) for line in (
            stat_line(100, "runner", 1, 100, 1000),            # the child (session leader)
            stat_line(101, "worker", 100, 100, 2000),          # its child
            stat_line(102, "grandchild", 101, 100, 4000),      # a grandchild
            stat_line(103, "orphan", 1, 100, 8000),            # reparented to init but still in the group
            stat_line(104, "detached", 101, 104, 16000),       # a descendant that started its own group
            stat_line(200, "unrelated", 1, 200, 999_999),      # not ours
            stat_line(201, "unrelated child", 200, 200, 999_999),
        )]
        self.assertEqual(proc.tree_rss_bytes(rows, 100, 4096), (1000 + 2000 + 4000 + 8000 + 16000) * 4096)
        self.assertEqual(proc.tree_rss_bytes(rows, 101, 1), 2000 + 4000 + 16000)   # a subtree, by parent id
        self.assertEqual(proc.tree_rss_bytes(rows, 9999, 4096), 0)
        self.assertEqual(proc.tree_rss_bytes([None, None], 100), 0)

    def test_read_process_table_reads_a_fixture_directory_and_skips_what_it_cannot_parse(self):
        with tempfile.TemporaryDirectory(dir=MODULE_BASE) as directory:
            root = Path(directory)
            for pid, line in ((10, stat_line(10, "a", 1, 10, 500)), (11, stat_line(11, "b", 10, 10, 700)),
                              (12, "not a stat line")):
                (root / str(pid)).mkdir()
                (root / str(pid) / "stat").write_text(line, encoding="utf-8")
            (root / "self").mkdir()                     # not a pid
            (root / "13").mkdir()                       # a process that vanished: no stat file
            rows = sorted(proc.read_process_table(str(root)))
            self.assertEqual([row[0] for row in rows], [10, 11])
            self.assertEqual(proc.tree_rss_bytes(rows, 10, 4096), 1200 * 4096)
            self.assertEqual(proc.read_process_table(str(root / "missing")), [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "needs a real /proc")
    def test_the_real_proc_of_this_process_parses(self):
        row = proc.parse_proc_stat(Path("/proc/self/stat").read_text(encoding="utf-8"))
        self.assertEqual(row[0], os.getpid())
        self.assertGreater(proc.sample_tree_rss(os.getpid()), 1_000_000)   # a Python process uses more than 1 MB

    def test_memory_exceeded_samples_at_most_twice_a_second_and_stays_true(self):
        clock = {"now": 1000.0}
        samples = []

        def sample(pid):
            samples.append(clock["now"])
            return (50 if clock["now"] == 1000.0 else 120) * 1024 * 1024

        child = proc.ChildProcess(types.SimpleNamespace(pid=4242, poll=lambda: None), None, 100 * 1024 * 1024)
        fake_time = types.SimpleNamespace(monotonic=lambda: clock["now"])
        with mock.patch.object(proc, "IS_WINDOWS", False), mock.patch.object(proc, "sample_tree_rss", sample), \
                mock.patch.object(proc, "time", fake_time):
            self.assertFalse(child.memory_exceeded())      # first call samples: 50 MB
            clock["now"] = 1000.2
            self.assertFalse(child.memory_exceeded())      # too soon, no new sample
            clock["now"] = 1000.6
            self.assertTrue(child.memory_exceeded())       # 120 MB is over 100 MB
            clock["now"] = 1001.5
            self.assertTrue(child.memory_exceeded())       # sticky, and no further sampling needed
            self.assertEqual(child.peak_memory_bytes(), 120 * 1024 * 1024)
        self.assertEqual(samples, [1000.0, 1000.6])

    def test_a_ceiling_of_zero_never_samples(self):
        child = proc.ChildProcess(types.SimpleNamespace(pid=4242, poll=lambda: None), None, 0)
        with mock.patch.object(proc, "IS_WINDOWS", False), \
                mock.patch.object(proc, "sample_tree_rss", side_effect=AssertionError("must not sample")):
            self.assertFalse(child.memory_exceeded())


class ChildMemoryCeiling(CoordinatorCase):
    """A worker tree that outgrows ``child_memory_mb`` is killed whole and its job fails, whichever lane it uses."""

    ALLOCATE = {"megabytes": 400, "step_megabytes": 20, "hold_seconds": 60}   # far above a 150 MB ceiling

    def assert_tree_gone(self, path: Path, with_allocator: bool = False) -> None:
        pids = self.read_pids(path)
        self.assertDead(pids["runner"])
        self.assertDead(pids["grandchild"])
        if with_allocator:
            self.assertDead(self.track(pids["allocator"]))

    def test_a_heavy_worker_over_the_ceiling_is_killed_with_its_tree_and_the_next_job_runs(self):
        coordinator = self.make(child_memory_mb=150)
        dataset = self.dataset(coordinator)
        earlier = self.submit(coordinator, dataset, "sleep:0")
        self.wait_state(coordinator, earlier, "succeeded")
        path = self.pid_file()
        started = time.monotonic()
        with self.assertRaises(JobError) as caught:
            coordinator.run_heavy("test_memory", {**self.ALLOCATE, "pid_file": str(path)}).wait(60)
        self.assertEqual(caught.exception.code, "MEMORY_LIMIT_EXCEEDED")
        self.assertLess(time.monotonic() - started, 40)   # killed promptly, not left to the 60 s hold
        self.assertNotIn(str(MODULE_BASE), json.dumps(caught.exception.to_dict()))
        self.assert_tree_gone(path)
        self.assertEqual(coordinator.stats()["heavy_running"], 0)
        # everything else is unaffected: the earlier job and its report, the dataset, new jobs, both lanes
        self.assertEqual(self.status(coordinator, earlier)["state"], "succeeded")
        self.assertEqual(coordinator.job_result(earlier["job_id"], earlier["capability"])[0], "succeeded")
        later = self.submit(coordinator, dataset, "sleep:0")
        self.assertEqual(self.wait_state(coordinator, later, "succeeded")["state"], "succeeded")
        self.assertEqual(coordinator.run_light("test_sleep", {"seconds": 0}), {"slept": 0})
        self.assertEqual(coordinator.run_heavy("test_sleep", {"seconds": 0}).wait(60), {"slept": 0})

    def test_memory_taken_by_a_grandchild_counts_against_the_same_ceiling(self):
        coordinator = self.make(child_memory_mb=150)
        path = self.pid_file()
        request = {**self.ALLOCATE, "pid_file": str(path), "allocator": "grandchild"}
        with self.assertRaises(JobError) as caught:
            coordinator.run_heavy("test_memory", request).wait(60)
        self.assertEqual(caught.exception.code, "MEMORY_LIMIT_EXCEEDED")
        self.assert_tree_gone(path, with_allocator=True)
        self.assertEqual(coordinator.run_heavy("test_sleep", {"seconds": 0}).wait(60), {"slept": 0})

    def test_a_v1_job_over_the_ceiling_ends_failed_with_the_code_and_leaves_no_files(self):
        coordinator = self.make(child_memory_mb=150)
        dataset = self.dataset(coordinator)
        job = self.submit(coordinator, dataset, "alloc:20")        # 400 MB in 20 MB steps
        status = self.wait_state(coordinator, job, "failed", 60)
        self.assertEqual(status["reason"]["code"], "MEMORY_LIMIT_EXCEEDED")
        self.assertEqual(coordinator.job_result(job["job_id"], job["capability"]), ("failed", None))
        self.assertFalse(coordinator.store.job_dir(job["job_id"]).exists())
        self.assertEqual(coordinator.stats()["heavy_running"], 0)
        again = self.submit(coordinator, dataset, "sleep:0")       # the dataset and the lane are fine
        self.wait_state(coordinator, again, "succeeded")

    def test_the_light_lane_has_the_same_ceiling(self):
        coordinator = self.make(child_memory_mb=150)
        path = self.pid_file()
        started = time.monotonic()
        with self.assertRaises(JobError) as caught:
            coordinator.run_light("test_memory", {**self.ALLOCATE, "pid_file": str(path)})
        self.assertEqual(caught.exception.code, "MEMORY_LIMIT_EXCEEDED")
        self.assertLess(time.monotonic() - started, 40)
        self.assert_tree_gone(path)
        self.assertEqual(coordinator.stats()["light_running"], 0)
        self.assertEqual(coordinator.run_light("test_sleep", {"seconds": 0}), {"slept": 0})   # the lane is free again
        stats = coordinator.stats()
        self.assertEqual((stats["heavy_running"], stats["light_running"], stats["jobs"]), (0, 0, 0), stats)
        self.assertLess(stats["stored_bytes"], 10_000)

    def test_a_worker_below_the_ceiling_and_a_ceiling_of_zero_are_not_disturbed(self):
        small = self.make(child_memory_mb=150)
        result = small.run_heavy("test_memory", {"megabytes": 60, "step_megabytes": 20, "hold_seconds": 0.5}).wait(60)
        self.assertEqual(result["allocated_bytes"], 60 * 1024 * 1024)
        off = self.make(child_memory_mb=0)
        result = off.run_light("test_memory", {"megabytes": 200, "step_megabytes": 50, "hold_seconds": 0.5})
        self.assertEqual(result["allocated_bytes"], 200 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
