"""Capabilities and opaque identifiers.

A capability is a 32-byte random URL-safe token. Only its SHA-256 digest is stored, and a presented token is
compared with ``hmac.compare_digest``. A missing, malformed, wrong or foreign token all fail the same way, so
the caller can answer them all with the same NOT_FOUND. Tokens are never logged or placed in error text.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

MAX_TOKEN_CHARACTERS = 256
_DUMMY_DIGEST = hashlib.sha256(b"fyd-no-such-capability").hexdigest()


def digest_of(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_capability() -> tuple[str, str]:
    """Return ``(token, digest)``. Hand out the token once; keep only the digest."""
    token = secrets.token_urlsafe(32)
    return token, digest_of(token)


def new_id(prefix: str) -> str:
    """Random opaque identifier such as ``ds_...`` or ``job_...`` (not guessable, not a secret)."""
    return f"{prefix}_{secrets.token_urlsafe(16)}"


def check_capability(presented, stored_digest: str | None) -> bool:
    """Constant-time check. Always does one digest comparison, whatever the input looks like."""
    if (not isinstance(presented, str) or not presented or len(presented) > MAX_TOKEN_CHARACTERS
            or stored_digest is None):
        hmac.compare_digest(_DUMMY_DIGEST, _DUMMY_DIGEST)
        return False
    try:
        candidate = digest_of(presented)
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(candidate, stored_digest)
