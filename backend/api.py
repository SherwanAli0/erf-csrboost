"""
api.py - FastAPI service for FixYourDataset: the /api/v1 interface and the legacy compatibility routes.

Run from the repository root:  python backend/api.py   (the PORT variable sets the port, default 7860).

One Uvicorn worker holds one in-process job coordinator (``backend/jobs``). Every task that touches data
(profiling, training, and the legacy analyze, score, convert and profile calls) runs in a child process the
coordinator starts, so a fit cannot block the event loop or the health routes, and a cancel or timeout kills
the whole process tree. This module only adapts HTTP to the coordinator: it owns no queue and no storage.

/api/v1 (bearer capabilities, typed errors, see docs/API.md):
  GET    /api/v1/version                      GET    /api/v1/samples
  POST   /api/v1/datasets                     DELETE /api/v1/datasets/{id}
  POST   /api/v1/jobs                         GET    /api/v1/jobs/{id}        DELETE /api/v1/jobs/{id}
  GET    /api/v1/jobs/{id}/result             GET    /api/v1/jobs/{id}/exports/{kind}
Health: GET and HEAD /healthz and /health. These handlers touch no lock and no disk.
Legacy (unchanged request and response shapes): GET / /index.html /api/samples /api/sample /stats /stats/<key>
/whoami/<key>; POST /api/profile /api/analyze /api/score /api/convert.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import hmac
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fastapi import APIRouter, FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.datastructures import Headers, MutableHeaders  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402
from starlette.requests import ClientDisconnect  # noqa: E402
from starlette.responses import Response  # noqa: E402

from jobs import Coordinator, JobConfig, JobError, states  # noqa: E402
from schemas import (DatasetCreated, Deleted, ErrorBody, ErrorResponse, JobAccepted, JobRequest,  # noqa: E402
                     validation_details)
from services.errors import ServiceError  # noqa: E402

log = logging.getLogger("fyd.api")

ROOT = HERE.parent
INDEX = ROOT / "frontend" / "index.html"
SAMPLES = ROOT / "samples"

APP_VERSION = "2.0.0-dev"
REPORT_SCHEMAS = {"training": "training-1.1", "profile": "app-profile-1.0", "job_config": "fyd-job-config-1.0"}

# Same metadata as backend/server.py (a test compares the two so they cannot drift apart).
SAMPLE_META = {
    "liver_disease.csv":   {"category": "Medical",        "emoji": "\U0001FA7A",
                            "label": "Liver disease",     "desc": "find the less common group: records without liver disease (583 rows, 2.5:1)"},
    "credit_default.csv":  {"category": "Finance",        "emoji": "\U0001F4B3",
                            "label": "Credit-card default", "desc": "predict who defaults next month (6,000 rows, 3.5:1)"},
    "student_dropout.csv": {"category": "Education",      "emoji": "\U0001F393",
                            "label": "Student dropout",   "desc": "find the less common group: currently enrolled students (4,424 rows, 4.6:1)"},
    "car_quality.csv":     {"category": "Manufacturing",  "emoji": "\U0001F697",
                            "label": "Car quality",       "desc": "catch the rare 'good' cars (1,728 rows, 24:1)"},
}
SAMPLE_ORDER = ["Medical", "Finance", "Education", "Manufacturing"]

LEGACY_POST_PATHS = ("/api/profile", "/api/analyze", "/api/score", "/api/convert")
LEGACY_BODY_LIMIT = 45 * 1024 * 1024    # as in backend/server.py: a 30 MB file is about 40 MB of base64 JSON
LEGACY_TOO_LARGE = "That request is too large. Please upload a file under 30 MB."
LEGACY_INVALID_SIZE = "That request has an invalid size."
LEGACY_TARGET_MISSING = "Could not read that file well enough to find a target column. Check the file's format."
JOB_BODY_LIMIT = 128 * 1024
SMALL_BODY_LIMIT = 64 * 1024
STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
HEAVY_POLL_SECONDS = 0.1
NO_REFERRER_EXEMPT = ("/", "/index.html")


# ------------------------------------------------------------------------------------------ errors

class ApiError(Exception):
    """A typed refusal raised by this module. ``message`` and ``details`` are fixed texts, safe to return."""

    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, dict(details or {})


class BodyTooLarge(Exception):
    """The request body went past the limit for its route while it was being received."""

    def __init__(self, limit: int):
        super().__init__("request body too large")
        self.limit = limit


def _escape_json_text(text: str) -> str:
    # "<", ">" and "&" never occur outside a JSON string, so writing them as \uXXXX keeps the JSON equal
    # while the bytes can never hold markup if a client or a browser mistakes them for HTML.
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def dumps_safe(value) -> bytes:
    """Strict JSON (no NaN), ASCII only, with markup characters escaped."""
    return _escape_json_text(json.dumps(value, allow_nan=False, separators=(",", ":"))).encode("ascii")


def escape_json_bytes(data: bytes) -> bytes:
    return data.replace(b"<", b"\\u003c").replace(b">", b"\\u003e").replace(b"&", b"\\u0026")


def json_response(value, status: int = 200, headers: dict | None = None) -> Response:
    return Response(content=dumps_safe(value), status_code=status, media_type="application/json", headers=headers)


def text_response(body: str, content_type: str, status: int = 200, headers: dict | None = None,
                  background=None) -> Response:
    merged = {"content-type": content_type, **(headers or {})}
    return Response(content=body.encode("utf-8"), status_code=status, headers=merged, background=background)


def legacy_json(status: int, value, headers: dict | None = None) -> Response:
    """The legacy wire format: plain ``json.dumps`` (NaN allowed, as the old server wrote it)."""
    return Response(content=json.dumps(value).encode("utf-8"), status_code=status,
                    headers={"content-type": "application/json", **(headers or {})})


def _retry_after(status: int, details: dict) -> dict:
    retry = details.get("retry_after_seconds") if isinstance(details, dict) else None
    if status in (429, 503) and isinstance(retry, (int, float)) and not isinstance(retry, bool) and retry > 0:
        return {"Retry-After": str(max(1, int(retry)))}
    return {}


def is_legacy_post(path: str) -> bool:
    return path in LEGACY_POST_PATHS


def refusal(request_id: str, path: str, status: int, code: str, message: str, details: dict | None = None,
            headers: dict | None = None) -> Response:
    """One typed error body for /api/v1 and everything else; the legacy POST routes keep ``{"ok": false, "error"}``."""
    details = dict(details or {})
    extra = {**(headers or {}), **_retry_after(status, details)}
    if is_legacy_post(path):
        return legacy_json(status, {"ok": False, "error": message}, extra)
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details, request_id=request_id))
    return json_response(body.model_dump(), status, extra)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


# ------------------------------------------------------------------------------------------ middleware

def body_limit(method: str, path: str, config: JobConfig) -> int:
    if method == "POST" and path == "/api/v1/datasets":
        return config.upload_limit_bytes
    if method == "POST" and path == "/api/v1/jobs":
        return JOB_BODY_LIMIT
    if method == "POST" and is_legacy_post(path):
        return LEGACY_BODY_LIMIT
    return SMALL_BODY_LIMIT


def _host_key(netloc: str, scheme: str) -> str:
    netloc = netloc.strip().lower()
    for default in (("http", ":80"), ("https", ":443")):
        if scheme == default[0] and netloc.endswith(default[1]):
            return netloc[: -len(default[1])]
    return netloc


def origin_allowed(headers: Headers) -> bool:
    """No Origin header is fine (curl, server to server). A present Origin must name this same host."""
    origin = headers.get("origin")
    if origin is None:
        return True
    parts = urlsplit(origin)
    if not parts.scheme or not parts.netloc:
        return False
    return _host_key(parts.netloc, parts.scheme.lower()) == _host_key(headers.get("host", ""), parts.scheme.lower())


class GuardMiddleware:
    """Request id, same-origin guard, body size limit while streaming, security headers, no-store.

    Written as a plain ASGI middleware so the body is counted as it arrives (a chunked upload with no
    Content-Length is stopped too). Anything that escapes the routes is logged by class name only and answered
    with a typed 500, so no traceback reaches a client or a log line.
    """

    def __init__(self, app, config: JobConfig):
        self.app = app
        self.config = config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = secrets.token_hex(8)
        scope.setdefault("state", {})["request_id"] = request_id
        method, path = scope["method"], scope["path"]
        headers = Headers(scope=scope)
        started = {"value": False}
        no_referrer = path not in NO_REFERRER_EXEMPT

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                started["value"] = True
                out = MutableHeaders(scope=message)
                out["X-Request-ID"] = request_id
                out["X-Content-Type-Options"] = "nosniff"
                out["X-Frame-Options"] = "DENY"
                if no_referrer:
                    out["Referrer-Policy"] = "no-referrer"
                if "cache-control" not in out:
                    out["Cache-Control"] = "no-store"
            await send(message)

        async def reply(status, code, message, details=None, extra=None):
            response = refusal(request_id, path, status, code, message, details, extra)
            await response(scope, receive, send_wrapper)

        if method in STATE_CHANGING and not origin_allowed(headers):
            await reply(403, "ORIGIN_NOT_ALLOWED", "This request came from a different origin.")
            return
        limit = body_limit(method, path, self.config)
        declared = headers.get("content-length")
        if declared is not None:
            if not declared.isdigit() or len(declared) > 15:
                await reply(400, "INVALID_CONTENT_LENGTH", LEGACY_INVALID_SIZE if is_legacy_post(path)
                            else "The Content-Length header is not valid.")
                return
            if int(declared) > limit:
                await too_large(reply, path, limit)
                return
        received = {"bytes": 0}

        async def counting_receive():
            message = await receive()
            if message["type"] == "http.request":
                received["bytes"] += len(message.get("body", b""))
                if received["bytes"] > limit:
                    raise BodyTooLarge(limit)
            return message

        try:
            await self.app(scope, counting_receive, send_wrapper)
        except BodyTooLarge:
            if not started["value"]:
                await too_large(reply, path, limit)
        except ClientDisconnect:   # the caller went away while sending: not a fault of the service
            log.info("client disconnected request=%s", request_id)
        except Exception as exc:  # noqa: BLE001 - the boundary: class name only, never the text or a traceback
            log.error("unhandled error class=%s request=%s", type(exc).__name__, request_id)
            if not started["value"]:
                await reply(500, "INTERNAL_ERROR", "The service could not complete this request.")


async def too_large(reply, path: str, limit: int) -> None:
    if is_legacy_post(path):   # the old server answered 400 with this text; the shape stays the same
        await reply(400, "REQUEST_TOO_LARGE", LEGACY_TOO_LARGE)
    else:
        await reply(413, "REQUEST_TOO_LARGE", "The request body is larger than the service allows.",
                    {"limit_bytes": limit})


# ------------------------------------------------------------------------------------------ helpers

router = APIRouter()


def coordinator_of(request: Request) -> Coordinator:
    return request.app.state.coordinator


def bearer_token(request: Request) -> str:
    """The capability from ``Authorization: Bearer``. Anything else counts as no token (the coordinator answers
    a missing, wrong or foreign token exactly like an unknown id)."""
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


async def read_body(request: Request) -> bytes:
    chunks = []
    async for chunk in request.stream():
        chunks.append(chunk)
    return b"".join(chunks)


def require_media_type(request: Request, allowed: tuple) -> None:
    media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media not in allowed:
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "This route does not accept that Content-Type.",
                       {"allowed": list(allowed)})


def _reject_constant(token):
    raise ValueError(token)


def parse_json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ApiError(400, "INVALID_JSON", "The request body is not valid JSON.") from None


# ------------------------------------------------------------------------------------------ health and info

@router.api_route("/healthz", methods=["GET", "HEAD"])
@router.api_route("/health", methods=["GET", "HEAD"])
async def healthz():
    """No lock, no disk, no coordinator: answers in constant time whatever the job lane is doing."""
    return text_response("ok", "text/plain")


@router.get("/api/v1/version")
async def version(request: Request):
    return json_response(request.app.state.version_info)


def _samples_body() -> dict:
    items = [{"name": name, **SAMPLE_META[name]} for name in SAMPLE_META if (SAMPLES / name).exists()]
    return {"samples": items, "order": SAMPLE_ORDER}


@router.get("/api/v1/samples")
def samples_v1():
    return json_response(_samples_body())


# ------------------------------------------------------------------------------------------ /api/v1 datasets, jobs

@router.post("/api/v1/datasets")
async def create_dataset(request: Request):
    require_media_type(request, ("application/octet-stream", "text/csv"))
    raw = await read_body(request)
    filename = request.headers.get("x-filename", "")[:300]
    created = await run_in_threadpool(coordinator_of(request).create_dataset, raw, filename)
    return json_response(DatasetCreated(**created).model_dump(), 201)


@router.delete("/api/v1/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, request: Request):
    coordinator_of(request).delete_dataset(dataset_id, bearer_token(request))
    return json_response(Deleted().model_dump())


@router.post("/api/v1/jobs")
async def create_job(request: Request):
    require_media_type(request, ("application/json",))
    data = parse_json(await read_body(request))
    try:
        model = JobRequest.model_validate(data)
    except ValidationError as exc:
        raise ApiError(422, "REQUEST_INVALID", "The job request is not valid.",
                       validation_details(exc.errors())) from None
    accepted = await run_in_threadpool(coordinator_of(request).submit_job, model.dataset_id, bearer_token(request),
                                       model.job_parameters())
    poll_url = f"/api/v1/jobs/{accepted['job_id']}"
    body = JobAccepted(job_id=accepted["job_id"], capability=accepted["capability"], state=accepted["state"],
                       poll_url=poll_url)
    return json_response(body.model_dump(), 202, {"Location": poll_url})


@router.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str, request: Request):
    return json_response(coordinator_of(request).job_status(job_id, bearer_token(request)))


@router.delete("/api/v1/jobs/{job_id}")
def cancel_job(job_id: str, request: Request):
    return json_response(coordinator_of(request).cancel_job(job_id, bearer_token(request)))


@router.get("/api/v1/jobs/{job_id}/result")
def job_result(job_id: str, request: Request):
    """200 report (succeeded or partial), 202 status (queued or running), 409 (no report will exist), 410 (expired)."""
    coordinator, token = coordinator_of(request), bearer_token(request)
    status = None
    for _ in range(3):   # the job can change state between the two calls
        state, report = coordinator.job_result(job_id, token)
        if state in states.RESULT_STATES:
            return json_response(report)
        if state == states.EXPIRED:
            raise ApiError(410, "JOB_EXPIRED", "This job has expired and its files were deleted.")
        status = coordinator.job_status(job_id, token)
        if state in states.LIVE_STATES:
            if status["state"] in states.LIVE_STATES:
                return json_response(status, 202, {"Retry-After": "2"})
            continue
        raise ApiError(409, "RESULT_NOT_AVAILABLE", f"The job is {state}; there is no report to return.",
                       {"state": state, "reason": status.get("reason"), "models": status.get("models", [])})
    return json_response(status, 202, {"Retry-After": "2"})


@router.get("/api/v1/jobs/{job_id}/exports/{kind}")
def job_export(job_id: str, kind: str, request: Request):
    data, content_type, filename = coordinator_of(request).job_export(job_id, bearer_token(request), kind)
    headers = {"content-type": content_type, "Content-Disposition": f'attachment; filename="{filename}"',
               "Cache-Control": "no-store"}
    if content_type.startswith("application/json"):
        data = escape_json_bytes(data)
    else:
        headers["Content-Security-Policy"] = ("default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
                                              "form-action 'none'; sandbox")
    return Response(content=data, headers=headers)


# ------------------------------------------------------------------------------------------ legacy: tracking

def track_module():
    """Imported on first use so importing this module (or serving /api/v1 only) never opens the usage database.

    ``track.py`` fixes its database path when it is imported (``/data`` if that directory exists, else
    ``data_local`` next to the repository). ``FYD_USAGE_DB`` names another file, which keeps a test or a trial run
    away from the real usage log. Unset, nothing changes."""
    import track
    override = os.environ.get("FYD_USAGE_DB", "").strip()
    if override and track.DB_PATH != override:
        track.DB_PATH = override
    return track


def trust_x_forwarded_for() -> bool:
    return os.environ.get("FYD_TRUST_X_FORWARDED_FOR", "").strip() == "1"


def trust_fly_client_ip() -> bool:
    """``Fly-Client-IP`` is set by the Fly proxy, which overwrites any value a caller sends. Anywhere else a caller
    can send it, so it is trusted only when ``FLY_APP_NAME`` is set (Fly sets it in its machines) or when
    ``FYD_TRUST_FLY_CLIENT_IP=1`` says so."""
    return (bool(os.environ.get("FLY_APP_NAME", "").strip())
            or os.environ.get("FYD_TRUST_FLY_CLIENT_IP", "").strip() == "1")


def visitor_details(request: Request) -> tuple:
    """(ip, user agent, referrer). The ip only feeds the daily pseudonymous hash in track.py and is not stored.

    ``Fly-Client-IP`` is trusted only on Fly (see ``trust_fly_client_ip``). ``X-Forwarded-For`` is ignored unless
    ``FYD_TRUST_X_FORWARDED_FOR=1``, because any client can send it. Otherwise the socket peer is used.
    """
    headers = request.headers
    ip = (headers.get("fly-client-ip") or "") if trust_fly_client_ip() else ""
    if not ip and trust_x_forwarded_for():
        ip = headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not ip:
        ip = request.client.host if request.client else ""
    referrer = headers.get("referer", "")
    if re.search(r"/(?:whoami|stats)/", referrer):   # such a URL carries the owner key: never store it
        referrer = ""
    return ip, headers.get("user-agent", ""), referrer


def is_owner_browser(request: Request) -> bool:
    cookie = request.cookies.get("fyd_owner")
    if not cookie:
        return False
    return hmac.compare_digest(cookie.encode("utf-8"), track_module().owner_token().encode("utf-8"))


def record_event(request: Request, kind: str, detail: str = "") -> None:
    ip, agent, referrer = visitor_details(request)
    track_module().log_event(kind, detail, ip, agent, referrer, is_owner=is_owner_browser(request))


def stats_key_valid(candidate: str) -> bool:
    expected = track_module().STATS_KEY
    return bool(expected and candidate) and hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


# ------------------------------------------------------------------------------------------ legacy: pages

def _not_found() -> Response:
    return text_response("not found", "text/plain", 404)


@router.api_route("/", methods=["GET", "HEAD"])
@router.api_route("/index.html", methods=["GET", "HEAD"])
def index_page(request: Request):
    page = INDEX.read_text(encoding="utf-8")
    task = None
    if request.method != "HEAD":   # uptime probes must not count as visits
        task = BackgroundTask(record_event, request, "visit")
    return text_response(page, "text/html; charset=utf-8", background=task)


@router.api_route("/api/samples", methods=["GET", "HEAD"])
def samples_legacy():
    return legacy_json(200, _samples_body())


@router.api_route("/api/sample", methods=["GET", "HEAD"])
def sample_legacy(request: Request):
    names = request.query_params.getlist("name")
    name = names[0] if names else ""
    path = SAMPLES / name
    if name in SAMPLE_META and path.exists():
        return text_response(path.read_text(encoding="utf-8"), "text/csv")
    return _not_found()


@router.api_route("/stats", methods=["GET", "HEAD"])
@router.api_route("/stats/{key:path}", methods=["GET", "HEAD"])
def stats_page(request: Request):
    """Owner page. The key may come as ``X-Stats-Key``, ``?key=`` or ``/stats/<key>``; it is compared in constant
    time and never logged. Without a valid key the answer is the same 404 as any unknown path."""
    key = (request.headers.get("x-stats-key") or request.query_params.get("key")
           or str(request.path_params.get("key", "")).strip("/"))
    if stats_key_valid(key):
        return text_response(track_module().stats_html(), "text/html; charset=utf-8")
    return _not_found()


@router.api_route("/whoami/{key:path}", methods=["GET", "HEAD"])
def whoami_page(request: Request):
    key = request.headers.get("x-stats-key") or str(request.path_params.get("key", "")).strip("/")
    if not stats_key_valid(key):
        return _not_found()
    cookie = f"fyd_owner={track_module().owner_token()}; Max-Age=31536000; Path=/; SameSite=Lax; HttpOnly"
    page = ("<!doctype html><meta charset=utf-8><body style='font:16px system-ui;"
            "max-width:520px;margin:60px auto;padding:0 16px'><h2>This browser is now "
            "marked as you.</h2><p>Your visits and runs will be tagged \"(you)\" and "
            "kept out of the visitor counts. Repeat this on any other device you use. "
            "To undo, clear this site's cookies.</p><p><a href='/'>Go to the app</a></p>")
    return text_response(page, "text/html; charset=utf-8", headers={"Set-Cookie": cookie})


# ------------------------------------------------------------------------------------------ legacy: POST routes

def _pick(body: dict, keys: tuple) -> dict:
    return {key: body[key] for key in keys if key in body}


async def wait_for_heavy(request: Request, handle) -> dict:
    """Wait for a legacy heavy job without blocking the event loop (short polls). The job is cancelled, and its
    process tree killed, when the client disconnects or the request task is cancelled."""
    try:
        while handle.state in states.LIVE_STATES:
            if await request.is_disconnected():
                await cancel_heavy(handle)
                raise ClientGone()
            await asyncio.sleep(HEAVY_POLL_SECONDS)
        return await run_in_threadpool(handle.wait, 5)
    except asyncio.CancelledError:
        await cancel_heavy(handle)
        raise


async def cancel_heavy(handle) -> None:
    """Cancel in a worker thread, shielded so a second cancellation cannot abandon it half way."""
    async def run():
        with suppress(Exception):
            await run_in_threadpool(handle.cancel)
        with suppress(Exception):
            await run_in_threadpool(handle.wait, 1)   # collect the outcome so the job record is dropped
    await asyncio.shield(asyncio.ensure_future(run()))


class ClientGone(Exception):
    """The client disconnected while its legacy job was running (the job has been cancelled)."""


async def legacy_post(request: Request):
    path = request.scope["path"]
    coordinator = coordinator_of(request)
    try:
        raw = await read_body(request)
        try:
            body = json.loads(raw or b"{}")
        except ValueError as exc:
            return legacy_json(400, {"ok": False, "error": str(exc)})
        if not isinstance(body, dict):
            return legacy_json(400, {"ok": False, "error": "The request body must be a JSON object."})
        if path == "/api/profile":
            result = await run_in_threadpool(coordinator.run_light, "legacy_profile", _pick(body, ("csv",)))
        elif path == "/api/convert":
            result = await run_in_threadpool(coordinator.run_light, "legacy_convert", _pick(body, ("b64", "name")))
        elif path == "/api/analyze":
            if not body.get("target"):
                return legacy_json(400, {"ok": False, "error": LEGACY_TARGET_MISSING})
            await run_in_threadpool(record_event, request, "analyze",
                                    f"{body.get('src', '?')} folds={body.get('folds', 1)}")
            handle = await run_in_threadpool(coordinator.run_heavy, "legacy_analyze", _pick(
                body, ("csv", "target", "method", "do_classify", "classifier", "folds")))
            result = await wait_for_heavy(request, handle)
        else:
            await run_in_threadpool(record_event, request, "score")
            handle = await run_in_threadpool(coordinator.run_heavy, "legacy_score", _pick(
                body, ("train_csv", "target", "method", "classifier", "score_csv")))
            result = await wait_for_heavy(request, handle)
        return legacy_json(200, result)
    except BodyTooLarge:
        raise
    except ClientGone:
        return legacy_json(499, {"ok": False, "error": "The request was cancelled."})
    except JobError as exc:
        return legacy_json(exc.http_status, {"ok": False, "error": exc.message}, _retry_after(exc.http_status, exc.details))
    except ServiceError as exc:
        return legacy_json(400, {"ok": False, "error": str(exc)})


for _path in LEGACY_POST_PATHS:
    router.add_api_route(_path, legacy_post, methods=["POST"])


# ------------------------------------------------------------------------------------------ app factory

async def api_error_handler(request: Request, exc: ApiError):
    return refusal(_request_id(request), request.scope["path"], exc.status, exc.code, exc.message, exc.details)


async def job_error_handler(request: Request, exc: JobError):
    return refusal(_request_id(request), request.scope["path"], exc.http_status, exc.code, exc.message, exc.details)


async def service_error_handler(request: Request, exc: ServiceError):
    status = {"APP_INTERNAL_ERROR": 500, "APP_PACKAGE_UNAVAILABLE": 503, "APP_ENGINE_UNAVAILABLE": 503}.get(exc.code, 422)
    return refusal(_request_id(request), request.scope["path"], status, exc.code, str(exc), exc.details)


async def body_too_large_handler(request: Request, exc: BodyTooLarge):
    path = request.scope["path"]
    if is_legacy_post(path):
        return refusal(_request_id(request), path, 400, "REQUEST_TOO_LARGE", LEGACY_TOO_LARGE)
    return refusal(_request_id(request), path, 413, "REQUEST_TOO_LARGE",
                   "The request body is larger than the service allows.", {"limit_bytes": exc.limit})


async def validation_error_handler(request: Request, exc: RequestValidationError):
    return refusal(_request_id(request), request.scope["path"], 422, "REQUEST_INVALID",
                   "The request is not valid.", validation_details(exc.errors()))


async def http_error_handler(request: Request, exc: StarletteHTTPException):
    path = request.scope["path"]
    if not path.startswith("/api/v1/"):   # the legacy server answered every unknown path with this text
        return _not_found()
    status = exc.status_code if exc.status_code in (404, 405) else 400
    code = "NOT_FOUND" if status == 404 else "METHOD_NOT_ALLOWED" if status == 405 else "BAD_REQUEST"
    message = {404: "Not found.", 405: "That method is not allowed on this route."}.get(status, "Bad request.")
    headers = {"Allow": exc.headers["Allow"]} if status == 405 and exc.headers and "Allow" in exc.headers else None
    return refusal(_request_id(request), path, status, code, message, headers=headers)


def _release_id() -> str:
    value = os.environ.get("FYD_RELEASE_ID", "").strip()
    return value if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value) else "unknown"


def _package_version() -> str:
    try:
        return importlib.metadata.version("rare_event_eval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the coordinator (it wipes its private storage root) and stop it on shutdown: every child process
    tree is killed and the storage root is deleted. A coordinator that is already running is left to its owner."""
    coordinator = app.state.coordinator
    started_here = not coordinator.stats()["started"]
    if started_here:
        await run_in_threadpool(coordinator.start)
    try:
        yield
    finally:
        if started_here:
            await run_in_threadpool(coordinator.stop)


def create_app(config: JobConfig | None = None, coordinator: Coordinator | None = None) -> FastAPI:
    """Build the app. Nothing starts until the server (or a test client) runs the lifespan."""
    if coordinator is not None:
        config = coordinator.config
    config = config or JobConfig.from_env()
    coordinator = coordinator or Coordinator(config)
    app = FastAPI(title="FixYourDataset service", version=APP_VERSION, lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None, redirect_slashes=False)
    app.state.config = config
    app.state.coordinator = coordinator
    app.state.version_info = {"app_version": APP_VERSION, "package_version": _package_version(),
                              "report_schemas": dict(REPORT_SCHEMAS), "release_id": _release_id()}
    app.include_router(router)
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(JobError, job_error_handler)
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(BodyTooLarge, body_too_large_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_middleware(GuardMiddleware, config=config)
    return app


_app = None


def __getattr__(name: str):
    """``api.app`` is created on first access, so importing this module starts nothing."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(create_app(), host="0.0.0.0", port=port, workers=1, access_log=False, timeout_keep_alive=5,
                limit_concurrency=64, server_header=False, timeout_graceful_shutdown=10, proxy_headers=False)


if __name__ == "__main__":
    main()
