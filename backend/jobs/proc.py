"""Child processes in their own process group or job, and whole-tree termination.

POSIX: the child is a session leader (``start_new_session``), so ``killpg`` reaches every descendant that
stayed in the group. Windows: the child is created suspended, put into a Job Object with
``KILL_ON_JOB_CLOSE`` and then resumed, so no descendant can be spawned outside the job and the whole tree
dies with ``TerminateJobObject`` (also if the parent process itself is killed). ``taskkill /T /F`` is the
fallback when a job cannot be used. The venv launcher ``python.exe`` on Windows starts the real interpreter as
a child, which is one more reason the whole tree, not one pid, is what gets killed and reaped.

Memory ceiling. A child tree may be given a ceiling in bytes. Windows: it is the Job Object's job-wide memory
limit, so the operating system refuses allocations beyond it for the whole tree, and a completion port tells
the supervisor that the limit was hit (``ChildProcess.memory_exceeded``). POSIX: the supervisor samples the
summed resident memory of the child's process tree from ``/proc`` (at most every half second, standard library
only) and ``memory_exceeded`` turns true when the sum passes the ceiling; the caller then kills the tree. The
parsing of ``/proc`` text is in pure functions (``parse_proc_stat``, ``tree_rss_bytes``) so it can be tested with
fixture text on any platform.

Nothing in this module logs a command line, an environment variable or a path.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:  # pragma: no cover - exercised on Windows only
    import ctypes
    from ctypes import wintypes

    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_SUSPENDED = 0x00000004
    CREATE_NO_WINDOW = 0x08000000
    _SYNCHRONIZE = 0x00100000
    _QUERY_LIMITED = 0x1000
    _STILL_ACTIVE = 259
    _JOB_EXTENDED_LIMIT = 9
    _KILL_ON_JOB_CLOSE = 0x2000
    _JOB_MEMORY_LIMIT = 0x200
    _JOB_ASSOCIATE_PORT = 7
    _JOB_MSG_PROCESS_MEMORY_LIMIT = 9
    _JOB_MSG_JOB_MEMORY_LIMIT = 10

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMITS(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _EXTENDED_LIMITS(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMITS), ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _ASSOCIATE_PORT(ctypes.Structure):
        _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", wintypes.HANDLE)]

    _kernel32.CreateIoCompletionPort.restype = wintypes.HANDLE
    _kernel32.CreateIoCompletionPort.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.c_size_t, wintypes.DWORD]
    _kernel32.GetQueuedCompletionStatus.restype = wintypes.BOOL
    _kernel32.GetQueuedCompletionStatus.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
                                                    ctypes.POINTER(ctypes.c_size_t),
                                                    ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD]
    _kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                    ctypes.c_void_p]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetCurrentProcess.argtypes = []
    _ntdll.NtResumeProcess.restype = ctypes.c_long
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
else:
    CREATE_NEW_PROCESS_GROUP = CREATE_SUSPENDED = CREATE_NO_WINDOW = 0


# ------------------------------------------------------------------ liveness

def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists and has not exited (a zombie counts as gone)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if IS_WINDOWS:
        handle = _kernel32.OpenProcess(_QUERY_LIMITED, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied means it exists
        try:
            code = wintypes.DWORD()
            if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            _kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie has not been reaped but is not running
        with open(f"/proc/{pid}/stat", "rb") as handle:
            state = handle.read().rsplit(b")", 1)[1].split()[0]
        return state != b"Z"
    except (OSError, IndexError):
        return True


class ParentWatch:
    """Answers "is my parent still there?" for the child's watchdog (see ``runner.py``)."""

    def __init__(self, pid: int):
        self.pid = pid
        self._handle = None
        self._initial_ppid = os.getppid()
        if IS_WINDOWS:
            self._handle = _kernel32.OpenProcess(_SYNCHRONIZE | _QUERY_LIMITED, False, pid) or None

    def alive(self) -> bool:
        if IS_WINDOWS:
            if not self._handle:
                return False
            return _kernel32.WaitForSingleObject(self._handle, 0) != 0  # 0 means signaled, i.e. exited
        return os.getppid() == self._initial_ppid and pid_alive(self.pid)


# ------------------------------------------------------------------ Windows job objects

class _Job:  # pragma: no cover - Windows only
    def __init__(self, handle, port=None):
        self.handle = handle
        self.port = port
        self.exceeded = False

    def terminate(self) -> None:
        if self.handle:
            _kernel32.TerminateJobObject(self.handle, 1)

    def poll_memory_limit(self) -> bool:
        """True once the operating system reported that a process pushed the job past its memory limit."""
        if self.exceeded or not self.port:
            return self.exceeded
        while True:
            count, key, overlapped = wintypes.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
            if not _kernel32.GetQueuedCompletionStatus(self.port, ctypes.byref(count), ctypes.byref(key),
                                                       ctypes.byref(overlapped), 0):
                break
            if count.value in (_JOB_MSG_PROCESS_MEMORY_LIMIT, _JOB_MSG_JOB_MEMORY_LIMIT):
                self.exceeded = True
        return self.exceeded

    def peak_bytes(self) -> int:
        if not self.handle:
            return 0
        info = _EXTENDED_LIMITS()
        if not _kernel32.QueryInformationJobObject(self.handle, _JOB_EXTENDED_LIMIT, ctypes.byref(info),
                                                   ctypes.sizeof(info), None):
            return 0
        return int(info.PeakJobMemoryUsed)

    def close(self) -> None:
        if self.port:
            _kernel32.CloseHandle(self.port)
            self.port = None
        if self.handle:
            _kernel32.CloseHandle(self.handle)
            self.handle = None


def _create_kill_on_close_job(memory_limit_bytes: int = 0):  # pragma: no cover - Windows only
    handle = _kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None
    info = _EXTENDED_LIMITS()
    info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE | (_JOB_MEMORY_LIMIT if memory_limit_bytes > 0 else 0)
    info.JobMemoryLimit = max(0, int(memory_limit_bytes))
    if not _kernel32.SetInformationJobObject(handle, _JOB_EXTENDED_LIMIT, ctypes.byref(info), ctypes.sizeof(info)):
        _kernel32.CloseHandle(handle)
        return None
    port = None
    if memory_limit_bytes > 0:   # the notification is optional: without it the limit still holds
        port = _kernel32.CreateIoCompletionPort(-1, None, 0, 1) or None
        if port is not None:
            link = _ASSOCIATE_PORT(None, port)
            if not _kernel32.SetInformationJobObject(handle, _JOB_ASSOCIATE_PORT, ctypes.byref(link),
                                                     ctypes.sizeof(link)):
                _kernel32.CloseHandle(port)
                port = None
    return _Job(handle, port)


_SELF_JOB = None


def adopt_self_into_job() -> bool:
    """Child side: put this process in its own job so ``kill_own_tree`` reaches every descendant (Windows).

    Call before any descendant is spawned. A no-op on POSIX, where the session leader's group is enough.
    """
    global _SELF_JOB
    if not IS_WINDOWS or _SELF_JOB is not None:
        return _SELF_JOB is not None
    job = _create_kill_on_close_job()  # pragma: no cover
    if job is None:  # pragma: no cover
        return False
    if not _kernel32.AssignProcessToJobObject(job.handle, _kernel32.GetCurrentProcess()):  # pragma: no cover
        job.close()
        return False
    _SELF_JOB = job  # pragma: no cover
    return True  # pragma: no cover


def kill_own_tree() -> None:
    """Child side: kill this process and everything it started. Does not return."""
    if IS_WINDOWS:  # pragma: no cover
        if _SELF_JOB is not None:
            _SELF_JOB.terminate()
        _taskkill(os.getpid())
    else:
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            pass
    os._exit(1)


def _taskkill(pid: int) -> None:  # pragma: no cover - Windows only
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
                       timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


# ------------------------------------------------------------------ POSIX memory sampling (pure parsing plus a reader)

MEMORY_SAMPLE_SECONDS = 0.5


def parse_proc_stat(text: str):
    """``(pid, ppid, pgrp, rss_pages)`` from the text of ``/proc/<pid>/stat``, or None if it does not parse.

    The command name is in parentheses and may itself contain spaces or parentheses, so the line is split at
    the last closing parenthesis. After it: state (field 3), ppid (4), pgrp (5), session (6) ... rss (24).
    """
    try:
        head, _, tail = text.rpartition(")")
        pid = int(head.split(" ", 1)[0])
        fields = tail.split()
        return pid, int(fields[1]), int(fields[2]), int(fields[21])
    except (ValueError, IndexError):
        return None


def tree_rss_bytes(rows, root_pid: int, page_size: int = 4096) -> int:
    """Summed resident memory in bytes of ``root_pid``, every descendant (by parent id) and every process still in
    the process group ``root_pid``. ``rows`` is an iterable of ``parse_proc_stat`` results. Pages shared between
    processes are counted once per process, so a forked tree is over-counted rather than under-counted."""
    rows = [row for row in rows if row is not None]
    children: dict = {}
    for pid, ppid, _pgrp, _rss in rows:
        children.setdefault(ppid, []).append(pid)
    members, stack = {root_pid}, [root_pid]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in members:
                members.add(child)
                stack.append(child)
    members.update(pid for pid, _ppid, pgrp, _rss in rows if pgrp == root_pid)
    return sum(rss for pid, _ppid, _pgrp, rss in rows if pid in members) * page_size


def read_process_table(proc_dir: str = "/proc") -> list:
    """Parsed ``stat`` rows of every process that can be read under ``proc_dir`` (unreadable ones are skipped)."""
    rows = []
    try:
        names = os.listdir(proc_dir)
    except OSError:
        return rows
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc_dir, name, "stat"), "r", encoding="utf-8", errors="replace") as handle:
                rows.append(parse_proc_stat(handle.read()))
        except OSError:
            continue
    return [row for row in rows if row is not None]


def sample_tree_rss(root_pid: int, proc_dir: str = "/proc") -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        page_size = 4096
    return tree_rss_bytes(read_process_table(proc_dir), root_pid, page_size)


# ------------------------------------------------------------------ parent side

class ChildProcess:
    """A started child plus what is needed to kill its whole tree."""

    def __init__(self, popen: subprocess.Popen, job=None, memory_limit_bytes: int = 0):
        self.popen = popen
        self._job = job
        self._memory_limit = max(0, int(memory_limit_bytes))
        self._memory_exceeded = False
        self._memory_peak = 0
        self._last_sample = 0.0

    def memory_exceeded(self) -> bool:
        """True once the tree went past its memory ceiling (sticky). Cheap to call on every supervisor tick."""
        if self._memory_exceeded or self._memory_limit <= 0:
            return self._memory_exceeded
        if IS_WINDOWS:  # pragma: no cover
            if self._job is not None and self._job.poll_memory_limit():
                self._memory_exceeded = True
        else:
            now = time.monotonic()
            if now - self._last_sample >= MEMORY_SAMPLE_SECONDS:
                self._last_sample = now
                used = sample_tree_rss(self.popen.pid)
                self._memory_peak = max(self._memory_peak, used)
                if used > self._memory_limit:
                    self._memory_exceeded = True
        return self._memory_exceeded

    def peak_memory_bytes(self) -> int:
        """Largest tree memory seen so far (Windows: the job's peak commit; POSIX: largest sample). For tests and
        measurements; nothing in the service depends on it."""
        if IS_WINDOWS:  # pragma: no cover
            return self._job.peak_bytes() if self._job is not None else 0
        return self._memory_peak

    @property
    def pid(self) -> int:
        return self.popen.pid

    @property
    def returncode(self):
        return self.popen.returncode

    def poll(self):
        return self.popen.poll()

    def wait(self, timeout=None):
        return self.popen.wait(timeout=timeout)

    def kill_tree(self, wait_seconds: float = 10.0) -> bool:
        """Kill the root and every descendant, then reap the root. Returns True when the root is reaped."""
        if IS_WINDOWS:  # pragma: no cover
            if self._job is not None:
                self._job.terminate()
            elif self.popen.poll() is None:
                _taskkill(self.popen.pid)
        else:
            try:
                os.killpg(self.popen.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if self.popen.poll() is None:
            try:
                self.popen.kill()
            except OSError:
                pass
        try:
            self.popen.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            return False
        return True

    def cleanup(self, wait_seconds: float = 10.0) -> None:
        """After the root exited by itself: kill stragglers left in the group or job, release the job handle."""
        if self.popen.poll() is None:
            self.kill_tree(wait_seconds)
        elif IS_WINDOWS:  # pragma: no cover
            if self._job is not None:
                self._job.terminate()
        else:
            try:
                os.killpg(self.popen.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.release()

    def release(self) -> None:
        if self._job is not None:
            self._job.close()
            self._job = None


def start_child(argv, *, cwd, env, use_job: bool = True, memory_limit_bytes: int = 0) -> ChildProcess:
    """Start ``argv`` with no stdin, no stdout and no stderr, in its own group (POSIX) or job (Windows).

    ``memory_limit_bytes`` above zero sets the tree's memory ceiling (see the module text); zero means none."""
    common = dict(cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, close_fds=True)
    if not IS_WINDOWS:
        return ChildProcess(subprocess.Popen(argv, start_new_session=True, **common),
                            memory_limit_bytes=memory_limit_bytes)
    flags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | (CREATE_SUSPENDED if use_job else 0)  # pragma: no cover
    popen = subprocess.Popen(argv, creationflags=flags, **common)  # pragma: no cover
    job = None
    if use_job:  # pragma: no cover
        handle = int(popen._handle)
        try:
            candidate = _create_kill_on_close_job(memory_limit_bytes)
            if candidate is not None and _kernel32.AssignProcessToJobObject(candidate.handle, handle):
                job = candidate
            elif candidate is not None:
                candidate.close()
        finally:
            if _ntdll.NtResumeProcess(handle) != 0:
                try:
                    popen.kill()
                finally:
                    if job is not None:
                        job.close()
                raise OSError("could not resume the worker process")
    return ChildProcess(popen, job, memory_limit_bytes)  # pragma: no cover


def python_executable() -> str:
    return sys.executable
