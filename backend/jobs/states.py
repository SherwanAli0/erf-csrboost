"""Job states and the legal transitions between them.

queued -> running -> succeeded | partial | failed | cancelled | timed_out | interrupted
queued -> cancelled | timed_out | interrupted
expired is a retrieval state reached only from a terminal state by the lifetime sweep.
"""
from __future__ import annotations

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
PARTIAL = "partial"
FAILED = "failed"
CANCELLED = "cancelled"
TIMED_OUT = "timed_out"
INTERRUPTED = "interrupted"
EXPIRED = "expired"

ALL_STATES = (QUEUED, RUNNING, SUCCEEDED, PARTIAL, FAILED, CANCELLED, TIMED_OUT, INTERRUPTED, EXPIRED)
FINISHED_STATES = frozenset({SUCCEEDED, PARTIAL, FAILED, CANCELLED, TIMED_OUT, INTERRUPTED})
LIVE_STATES = frozenset({QUEUED, RUNNING})
RESULT_STATES = frozenset({SUCCEEDED, PARTIAL})

TRANSITIONS = {
    QUEUED: frozenset({RUNNING, CANCELLED, TIMED_OUT, INTERRUPTED}),
    RUNNING: frozenset({SUCCEEDED, PARTIAL, FAILED, CANCELLED, TIMED_OUT, INTERRUPTED}),
    SUCCEEDED: frozenset({EXPIRED}),
    PARTIAL: frozenset({EXPIRED}),
    FAILED: frozenset({EXPIRED}),
    CANCELLED: frozenset({EXPIRED}),
    TIMED_OUT: frozenset({EXPIRED}),
    INTERRUPTED: frozenset({EXPIRED}),
    EXPIRED: frozenset(),
}

# Progress stages written by the child process (never a time-based percentage).
STAGES = ("parsing", "profiling", "splitting", "fitting", "evaluating", "reporting", "done")


class IllegalTransition(RuntimeError):
    """Raised when code tries to move a job along an edge that the state machine does not have."""


def is_terminal(state: str) -> bool:
    """True for every state a job cannot leave except by expiry (and for expired itself)."""
    return state in FINISHED_STATES or state == EXPIRED


def can_transition(current: str, new: str) -> bool:
    return new in TRANSITIONS.get(current, frozenset())


def check_transition(current: str, new: str) -> str:
    """Return ``new`` if the edge is legal, otherwise raise ``IllegalTransition``."""
    if not can_transition(current, new):
        raise IllegalTransition(f"illegal job state transition {current} -> {new}")
    return new
