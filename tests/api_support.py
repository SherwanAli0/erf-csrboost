"""Shared helpers for ``test_api_v1.py`` and ``test_api_compat.py`` (not a test module itself).

Importing this module puts ``backend`` and the evaluation test helpers on ``sys.path``, silences the
Starlette deprecation notice about ``httpx`` in its test client, and provides: a synthetic mixed table, a log
capture for the whole root logger, a process scan that finds leftover children by a private directory name,
and a launcher for ``python backend/api.py`` on a free port.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import warnings

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
PACKAGE_SOURCE = ROOT / "evaluation" / "src"
PACKAGE_TESTS = ROOT / "evaluation" / "tests"

if importlib.util.find_spec("rare_event_eval") is None:  # child processes inherit PYTHONPATH
    sys.path.insert(0, str(PACKAGE_SOURCE))
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE_SOURCE), os.environ.get("PYTHONPATH", "")]))
for entry in (str(BACKEND), str(PACKAGE_TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

warnings.filterwarnings("ignore", message=".*httpx.*starlette.testclient.*")
warnings.filterwarnings("ignore", message=".*starlette.testclient.*")

from mixed_support import mixed_table, table_to_csv  # noqa: E402


def synthetic_csv(rows: int = 300, positives: int = 40, labels=("fraud", "ok")) -> bytes:
    """A small mixed table (numeric, categorical, missing cells) with a rare class in ``label``."""
    return table_to_csv(mixed_table(n=rows, positives=positives, labels=labels))


def job_request(dataset: dict, **extra) -> dict:
    return {"dataset_id": dataset["dataset_id"], "target": "label", "positive_label": "fraud",
            "negative_label": "ok", "seed": 7, **extra}


# ----------------------------------------------------------------------------------------- logs

class LogCapture(logging.Handler):
    """Collects every log record of the process (root logger, all levels) as formatted text."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list = []
        self._saved_level = None

    def emit(self, record):
        try:
            self.lines.append(f"{record.name} {record.levelname} {record.getMessage()}")
        except Exception:  # noqa: BLE001 - capturing must never break a test
            self.lines.append(f"{record.name} {record.levelname} <unformattable>")

    def __enter__(self):
        root = logging.getLogger()
        self._saved_level = root.level
        root.setLevel(logging.DEBUG)
        root.addHandler(self)
        return self

    def __exit__(self, *exc_info):
        root = logging.getLogger()
        root.removeHandler(self)
        root.setLevel(self._saved_level)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def service_text(self) -> str:
        """The log without the test client's own lines (httpx logs every request URL, including the one that
        carries an owner key in the path; the service itself has no access log and logs no URL)."""
        return "\n".join(line for line in self.lines if not line.startswith(("httpx ", "httpcore ")))


# ----------------------------------------------------------------------------------------- processes

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


def kill_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def wait_for(predicate, timeout: float = 30.0, step: float = 0.05, message: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    raise AssertionError(f"timed out waiting for {message}")


def remove_directory(path: Path) -> bool:
    from jobs import store
    return store.remove_tree(path)


def private_directory(prefix: str = "fyd-api-test-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix)).resolve()


# ----------------------------------------------------------------------------------------- real server

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerProcess:
    """``python -B backend/api.py`` on a free port with a private storage root.

    Output goes to a file inside ``base`` so a test can search it for secrets. ``stop`` asks for a clean
    shutdown (Ctrl+Break on Windows, SIGTERM elsewhere) and returns the exit code.
    """

    def __init__(self, base: Path, env_extra: dict | None = None):
        self.base = Path(base)
        self.port = free_port()
        self.root = self.base / "root"
        self.log_path = self.base / "server.log"
        env = {key: value for key, value in os.environ.items() if key != "STATS_KEY"}
        env.update({"PORT": str(self.port), "FYD_TEMP_ROOT": str(self.root), "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUTF8": "1"})
        env.update(env_extra or {})
        self.command = [sys.executable, "-B", str(BACKEND / "api.py")]
        self._log = open(self.log_path, "wb")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen(self.command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                        stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags)
        self.url = f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout: float = 60.0) -> None:
        import httpx

        def ready():
            if self.process.poll() is not None:
                raise AssertionError(f"the server exited early with code {self.process.returncode}: {self.output()[-800:]}")
            try:
                return httpx.get(self.url + "/healthz", timeout=2).status_code == 200
            except httpx.HTTPError:
                return False
        wait_for(ready, timeout, 0.2, "the server to answer /healthz")

    def output(self) -> str:
        if not self._log.closed:
            self._log.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    # Uvicorn finishes its shutdown and then re-raises the signal it caught, so the exit status of a clean stop is
    # 3 on Windows (Ctrl+Break) and minus the signal number elsewhere (SIGTERM), not 0.
    CLEAN_EXIT_CODES = (0, 3) if os.name == "nt" else (0, -signal.SIGTERM)

    def stopped_cleanly(self, code) -> bool:
        return code in self.CLEAN_EXIT_CODES and "Application shutdown complete" in self.output()

    def stop(self, timeout: float = 30.0) -> int | None:
        if self.process.poll() is None:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                kill_pid(self.process.pid)
                self.process.wait(10)
        self._log.close()
        return self.process.returncode


# ----------------------------------------------------------------------------------------- app fixtures

import httpx  # noqa: E402,F401
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from jobs import Coordinator, JobConfig  # noqa: E402


def strip_request_id(response) -> dict:
    body = response.json()
    body["error"] = {key: value for key, value in body["error"].items() if key != "request_id"}
    return body


class ApiCase(unittest.TestCase):
    """A real app and coordinator with a private storage root, log capture and an end-of-class leak check."""

    CONFIG: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.log = LogCapture()
        cls.log.__enter__()
        cls.base = private_directory()
        cls.marker = cls.base.name
        settings = dict(temp_root=str(cls.base / "root"), sweep_interval_seconds=0.2, upload_limit_bytes=1_000_000,
                        max_datasets=8)
        settings.update(cls.CONFIG)
        cls.coordinator = Coordinator(JobConfig(**settings))
        cls.app = api.create_app(coordinator=cls.coordinator)
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.client.__enter__()
        cls.secrets: list = []
        cls.jobs: list = []

    @classmethod
    def tearDownClass(cls):
        problems = []
        for job in cls.jobs:   # leave no live job behind, then check
            try:
                cls.client.delete(f"/api/v1/jobs/{job['job_id']}", headers=auth(job["capability"]))
            except Exception:  # noqa: BLE001
                pass
        cls.client.__exit__(None, None, None)
        leftover = processes_with(cls.marker)
        for pid in leftover:
            kill_pid(pid)
        if leftover:
            problems.append(f"{len(leftover)} child process(es) still running")
        if cls.coordinator.store.root.exists():
            problems.append("the job storage root was not deleted when the app stopped")
        log_text = cls.log.service_text
        cls.log.__exit__(None, None, None)
        leaked = [secret[:6] + "..." for secret in cls.secrets if secret in log_text]
        if leaked:
            problems.append(f"capabilities found in the log: {leaked}")
        remove_directory(cls.base)
        if cls.base.exists():
            problems.append("the private test directory could not be removed")
        if problems:
            raise AssertionError("; ".join(problems))

    # ------------------------------------------------------------ helpers
    def remember(self, *tokens) -> None:
        type(self).secrets.extend(token for token in tokens if token)

    def make_dataset(self, raw: bytes | None = None, name: str = "table.csv", content_type: str = "text/csv") -> dict:
        response = self.client.post("/api/v1/datasets", content=raw if raw is not None else synthetic_csv(),
                                    headers={"Content-Type": content_type, "X-Filename": name})
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.remember(body["capability"])
        self.addCleanup(lambda: self.client.delete(f"/api/v1/datasets/{body['dataset_id']}",
                                                   headers=auth(body["capability"])))
        return body

    def submit(self, dataset: dict, expect: int = 202, **extra):
        response = self.client.post("/api/v1/jobs", json=job_request(dataset, **extra),
                                    headers=auth(dataset["capability"]))
        self.assertEqual(response.status_code, expect, response.text)
        body = response.json()
        if expect == 202:
            self.remember(body["capability"])
            type(self).jobs.append(body)
        return body if expect == 202 else response

    def status(self, job: dict, token: str | None = None) -> dict:
        response = self.client.get(f"/api/v1/jobs/{job['job_id']}", headers=auth(token or job["capability"]))
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def wait_state(self, job: dict, wanted, timeout: float = 120.0) -> dict:
        wanted = (wanted,) if isinstance(wanted, str) else tuple(wanted)
        holder = {}

        def reached():
            holder["status"] = self.status(job)
            return holder["status"]["state"] in wanted
        wait_for(reached, timeout, 0.1, f"job state {wanted}")
        return holder["status"]

    def wait_fitting(self, job: dict, timeout: float = 120.0) -> dict:
        holder = {}

        def fitting():
            holder["status"] = self.status(job)
            return (holder["status"].get("progress") or {}).get("current_model")
        wait_for(fitting, timeout, 0.1, "a model to be fitted")
        return holder["status"]

    def cancel(self, job: dict):
        return self.client.delete(f"/api/v1/jobs/{job['job_id']}", headers=auth(job["capability"]))

    def assertNoSecretIn(self, text: str) -> None:
        for secret in self.secrets:
            self.assertNotIn(secret, text)


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


