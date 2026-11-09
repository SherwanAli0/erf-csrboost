"""Private temporary storage: one root, one directory per dataset and per job, byte accounting, cleanup.

The root is created with owner-only permissions where the platform supports them and holds a marker file.
``prepare_root`` deletes everything inside a root that carries the marker (a stale root of an earlier
process) and refuses to touch a non-empty directory that does not carry it, so a wrong configuration can
never wipe an unrelated folder. Nothing here pickles anything; JSON is written atomically (temp file in the
same directory, then ``os.replace``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import time

MARKER = ".fyd-jobs-root"
MARKER_TEXT = "FixYourDataset job storage. Everything in this directory is deleted when the service starts.\n"


class StoreError(RuntimeError):
    """Storage refused to do something (unsafe root, unreadable file). The text never holds a path."""


def _on_remove_error(function, path, excinfo):  # pragma: no cover - platform dependent
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(path)
    except OSError:
        pass


def remove_tree(path: Path, *, attempts: int = 8, pause: float = 0.25) -> bool:
    """Delete a directory tree. Retries because Windows keeps a file busy for a moment after a process exits.

    Returns True when the path no longer exists.
    """
    path = Path(path)
    for attempt in range(attempts):
        if not path.exists() and not path.is_symlink():
            return True
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path, onerror=_on_remove_error)
        except OSError:
            pass
        if not path.exists():
            return True
        time.sleep(pause)
    return not path.exists()


def tree_bytes(path: Path) -> int:
    """Total size of the regular files under ``path`` (symbolic links are not followed)."""
    total = 0
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def write_json_atomic(path: Path, value, *, allow_nan: bool = False) -> int:
    """Write JSON to ``path`` through a temp file and ``os.replace``. Returns the byte size."""
    path = Path(path)
    data = json.dumps(value, allow_nan=allow_nan).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:  # pragma: no cover
            pass
    os.replace(temporary, path)
    return len(data)


def read_json_bounded(path: Path, max_bytes: int):
    """Parse a JSON file written by a child process. Returns ``None`` if absent; raises ``StoreError`` if it is
    too large or not valid JSON (the file is untrusted input)."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > max_bytes:
        raise StoreError("file too large")
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        raise StoreError("file unreadable") from None


class Store:
    """The private storage root. Methods only build paths from identifiers this process generated."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.datasets = self.root / "datasets"
        self.jobs = self.root / "jobs"
        self.light = self.root / "light"

    # ------------------------------------------------------------ root
    def prepare_root(self) -> None:
        """Create a clean private root. Wipes a stale root that carries the marker."""
        root = self.root
        if root.is_symlink():
            raise StoreError("the storage root must not be a symbolic link")
        if root.exists():
            if not root.is_dir():
                raise StoreError("the storage root is not a directory")
            entries = list(root.iterdir())
            if entries and not (root / MARKER).is_file():
                raise StoreError("the storage root is not empty and is not a job storage root; refusing to wipe it")
            for entry in entries:
                if not remove_tree(entry):
                    raise StoreError("could not clear the storage root")
        else:
            root.mkdir(parents=True, mode=0o700)
        try:
            os.chmod(root, 0o700)
        except OSError:  # pragma: no cover - Windows ignores most of the mode
            pass
        (root / MARKER).write_text(MARKER_TEXT, encoding="utf-8")
        for directory in (self.datasets, self.jobs, self.light):
            directory.mkdir(mode=0o700)

    def destroy_root(self) -> bool:
        """Delete the whole root (used on shutdown). Only a root with the marker is removed."""
        if not (self.root / MARKER).is_file():
            return not self.root.exists()
        return remove_tree(self.root)

    # ------------------------------------------------------------ directories
    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets / dataset_id

    def job_dir(self, job_id: str) -> Path:
        return self.jobs / job_id

    def light_dir(self, job_id: str) -> Path:
        return self.light / job_id

    def make_dir(self, path: Path) -> Path:
        path.mkdir(mode=0o700)
        return path

    # ------------------------------------------------------------ accounting
    def used_bytes(self) -> int:
        return tree_bytes(self.root)

    def leftover_entries(self) -> list:
        """Names of every directory still present under datasets, jobs and light (for tests and stats)."""
        out = []
        for parent in (self.datasets, self.jobs, self.light):
            try:
                out.extend(entry.name for entry in parent.iterdir())
            except OSError:
                continue
        return out
