"""Bounded background jobs for the FixYourDataset service.

``Coordinator`` runs every data-touching task in a child process (see ``runner.py``), with one active heavy
job, one waiting slot, a small light lane, capability-based ownership and real process-tree cancellation.
Importing this package imports only the standard library; the evaluation package, pandas and the legacy
engine are used only inside the child processes.
"""
from .config import JobConfig
from .coordinator import Coordinator, HeavyHandle, JobError
from . import states

__all__ = ["Coordinator", "HeavyHandle", "JobConfig", "JobError", "states"]
