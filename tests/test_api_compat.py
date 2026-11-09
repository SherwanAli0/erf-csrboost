"""Behaviour tests for the legacy compatibility routes of ``backend/api.py``.

The routes must keep the wire format of ``backend/server.py``: the same keys as a direct call to the legacy
engine, errors as 400 ``{"ok": false, "error": ...}``, 404 ``not found`` as text, the same usage events. The
bundled sample ``liver_disease.csv`` is used for the real calls, and results are compared with the legacy engine
called in this process.

The usage database is redirected to a private temporary file and the owner key is a test value set on the
``track`` module (the module reads ``STATS_KEY`` once at import), so no real database is touched.

Run from the repository root::

    python -B -m unittest discover -s tests -p "test_api_compat.py" -v
"""
from __future__ import annotations

import asyncio
import base64
import gc
import json
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import time
import unittest
from unittest import mock

import api_support as support  # noqa: F401  (sets sys.path and warning filters before the imports below)
from api_support import ApiCase, LogCapture, auth, job_request, processes_with, synthetic_csv, wait_for

DATA_LOCAL = support.ROOT / "data_local"
DATA_LOCAL_EXISTED = DATA_LOCAL.exists()

import httpx  # noqa: E402

import api  # noqa: E402
import engine as legacy_engine  # noqa: E402
import server as legacy_server  # noqa: E402
import track  # noqa: E402

SAMPLE = support.ROOT / "samples" / "liver_disease.csv"
CSV_TEXT = SAMPLE.read_text(encoding="utf-8")
LINES = CSV_TEXT.splitlines()
SMALL_SCORE_CSV = "\n".join(LINES[:21]) + "\n"
TARGET = legacy_engine.profile(legacy_engine.read_csv(CSV_TEXT))["suggested_target"]
LONG_ANALYZE = {"csv": CSV_TEXT, "target": TARGET, "method": "smote", "do_classify": True,
                "classifier": "random_forest", "folds": 30, "src": "upload"}   # about two minutes if left alone

STATE: dict = {}


def setUpModule():
    base = support.private_directory("fyd-compat-test-")
    STATE.update(base=base, key="owner-" + secrets.token_urlsafe(18), calls=[],
                 saved=(track.DB_PATH, track.STATS_KEY, track.log_event))
    track.DB_PATH = str(base / "usage.db")
    track.STATS_KEY = STATE["key"]
    original = STATE["saved"][2]

    def recording(kind, detail="", ip="", user_agent="", referrer="", is_owner=0):
        STATE["calls"].append({"kind": kind, "detail": detail, "ip": ip, "agent": user_agent, "referrer": referrer,
                               "owner": bool(is_owner)})
        return original(kind, detail, ip, user_agent, referrer, is_owner=is_owner)
    track.log_event = recording


def tearDownModule():
    track.DB_PATH, track.STATS_KEY, track.log_event = STATE["saved"]
    gc.collect()   # track.py closes its sqlite connections only when they are collected, and Windows will not delete an open file
    support.remove_directory(STATE["base"])
    problems = []
    if STATE["base"].exists():
        problems.append("the private usage directory could not be removed")
    if not DATA_LOCAL_EXISTED and DATA_LOCAL.exists():   # importing track creates it; remove what the import made
        try:
            DATA_LOCAL.rmdir()
        except OSError:
            problems.append("data_local is not empty")
    if problems:
        raise AssertionError("; ".join(problems))


def usage_rows() -> list:
    connection = sqlite3.connect(track.DB_PATH)
    try:
        return connection.execute("SELECT kind, detail, is_owner FROM events ORDER BY ts, rowid").fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()


def dumps_legacy(value) -> str:
    return json.dumps(value)


class LegacyCase(ApiCase):
    def setUp(self):
        STATE["calls"].clear()

    def post(self, path: str, body, **kwargs):
        content = body if isinstance(body, (bytes, str)) else json.dumps(body)
        return self.client.post(path, content=content, headers={"Content-Type": "application/json"}, **kwargs)

    def assertLegacyError(self, response, status: int, message: str | None = None):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        self.assertEqual(set(body), {"ok", "error"})
        self.assertIs(body["ok"], False)
        if message is not None:
            self.assertEqual(body["error"], message)
        self.assertEqual(response.headers["content-type"], "application/json")
        for forbidden in ("Traceback", str(self.base), str(support.ROOT)):
            self.assertNotIn(forbidden, response.text)
        return body["error"]


# ============================================================================================ POST routes

class LegacyPost(LegacyCase):
    def test_profile_returns_what_the_engine_returns(self):
        response = self.post("/api/profile", {"csv": CSV_TEXT})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.headers["cache-control"], "no-store")
        direct = legacy_engine.profile(legacy_engine.read_csv(CSV_TEXT))
        self.assertEqual(set(response.json()), set(direct))
        self.assertEqual(response.json(), json.loads(json.dumps(direct)))
        self.assertEqual(response.json()["suggested_target"], TARGET)
        self.assertEqual(STATE["calls"], [])   # the old server logged nothing for a profile

    def test_convert_returns_the_csv_and_notes_of_the_engine(self):
        raw = SAMPLE.read_bytes()
        response = self.post("/api/convert", {"b64": base64.b64encode(raw).decode(), "name": "liver_disease.csv"})
        self.assertEqual(response.status_code, 200)
        text, notes = legacy_engine.any_to_csv(raw, "liver_disease.csv")
        self.assertEqual(response.json(), {"ok": True, "csv": text, "notes": notes})
        self.assertEqual(STATE["calls"], [])

    def test_analyze_relays_the_engine_result_and_logs_the_event_before_the_run(self):
        body = {"csv": CSV_TEXT, "target": TARGET, "method": "smote", "do_classify": False, "classifier": "logreg",
                "folds": 1, "src": "demo:liver_disease.csv"}
        response = self.post("/api/analyze", body)
        self.assertEqual(response.status_code, 200, response.text[:300])
        direct = legacy_engine.analyze(CSV_TEXT, TARGET, "smote", do_classify=False, classifier="logreg", folds=1)
        result = response.json()
        self.assertEqual(set(result), set(direct))
        self.assertEqual(dumps_legacy(result), dumps_legacy(direct))   # the engine's own answer, relayed
        self.assertEqual([(call["kind"], call["detail"]) for call in STATE["calls"]],
                         [("analyze", "demo:liver_disease.csv folds=1")])
        self.assertEqual(STATE["calls"][0]["ip"], "testclient")
        self.assertIn(("analyze", "demo:liver_disease.csv folds=1", 0), usage_rows())

    def test_score_returns_the_engine_result_and_logs_a_score_event(self):
        body = {"train_csv": CSV_TEXT, "target": TARGET, "method": "smote", "classifier": "logreg",
                "score_csv": SMALL_SCORE_CSV}
        response = self.post("/api/score", body)
        self.assertEqual(response.status_code, 200, response.text[:300])
        direct = legacy_engine.score_new(CSV_TEXT, TARGET, "smote", "logreg", SMALL_SCORE_CSV)
        self.assertEqual(set(response.json()), set(direct))
        self.assertEqual(dumps_legacy(response.json()), dumps_legacy(direct))
        self.assertEqual([(call["kind"], call["detail"]) for call in STATE["calls"]], [("score", "")])

    def test_errors_are_400_with_the_legacy_shape_and_texts(self):
        self.assertLegacyError(self.post("/api/profile", {}), 400, "The request is missing a required field, or names "
                               "a column that is not in the file: 'csv'.")
        self.assertLegacyError(self.post("/api/convert", {"name": "x.csv"}), 400, "The request is missing a required "
                               "field, or names a column that is not in the file: 'b64'.")
        try:
            json.loads("{bad")
        except ValueError as exc:
            expected = str(exc)
        self.assertLegacyError(self.post("/api/profile", "{bad"), 400, expected)
        self.assertLegacyError(self.post("/api/analyze", {"csv": CSV_TEXT}), 400,
                               "Could not read that file well enough to find a target column. Check the file's format.")
        self.assertEqual(STATE["calls"], [])   # refused before the event, as the old server did
        for status_body in ([1, 2], "3"):
            self.assertLegacyError(self.post("/api/profile", json.dumps(status_body)), 400)
        try:
            legacy_engine.analyze(CSV_TEXT, "no_such_column", "smote", do_classify=False, classifier="logreg", folds=1)
        except Exception as exc:  # noqa: BLE001 - the same call the live server makes
            expected = str(exc)
        self.assertLegacyError(self.post("/api/analyze", {"csv": CSV_TEXT, "target": "no_such_column", "folds": 1}),
                               400, expected)
        self.assertEqual([call["kind"] for call in STATE["calls"]], ["analyze"])   # failed runs are logged too
        self.assertLegacyError(self.post("/api/score", {"train_csv": CSV_TEXT, "target": TARGET}), 400, "The request is "
                               "missing a required field, or names a column that is not in the file: 'score_csv'.")
        try:
            legacy_engine.any_to_csv(b"not a workbook", "x.xlsx")
        except Exception as exc:  # noqa: BLE001
            expected = str(exc)
        self.assertLegacyError(self.post("/api/convert", {"b64": base64.b64encode(b"not a workbook").decode(),
                                                          "name": "x.xlsx"}), 400, expected)

    def test_unknown_paths_and_wrong_methods_answer_404_not_found_as_text(self):
        for response in (self.client.get("/nope"), self.client.get("/api/profile"), self.client.post("/api/nope", content=b"{}"),
                         self.client.get("/api/analyze"), self.client.delete("/api/profile"), self.client.get("/api/sample/x")):
            self.assertEqual((response.status_code, response.text, response.headers["content-type"]),
                             (404, "not found", "text/plain"))

    def test_a_declared_body_over_45_mb_is_refused_without_reading_it_in_the_legacy_shape(self):
        response = self.client.post("/api/profile", content=b"{}", headers={"Content-Type": "application/json",
                                                                          "Content-Length": str(45 * 1024 * 1024 + 1)})
        self.assertLegacyError(response, 400, "That request is too large. Please upload a file under 30 MB.")
        response = self.client.post("/api/convert", content=b"{}", headers={"Content-Type": "application/json",
                                                                          "Content-Length": "-4"})
        self.assertLegacyError(response, 400, "That request has an invalid size.")
        self.assertEqual(self.coordinator.stats()["light_waiting"], 0)

    def test_a_cross_origin_post_is_refused_in_the_legacy_shape(self):
        response = self.client.post("/api/profile", content=json.dumps({"csv": CSV_TEXT}),
                                    headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
        self.assertLegacyError(response, 403)
        same = self.client.post("/api/profile", content=json.dumps({"csv": CSV_TEXT}),
                                headers={"Content-Type": "application/json", "Origin": "http://testserver"})
        self.assertEqual(same.status_code, 200)

    def test_the_content_type_is_not_checked_on_legacy_routes_as_before(self):
        response = self.client.post("/api/profile", content=json.dumps({"csv": CSV_TEXT}))
        self.assertEqual(response.status_code, 200)
        response = self.client.post("/api/profile", content=json.dumps({"csv": CSV_TEXT}),
                                    headers={"Content-Type": "text/plain"})
        self.assertEqual(response.status_code, 200)


# ============================================================================================ pages and stats

class LegacyPages(LegacyCase):
    def test_the_index_page_and_head(self):
        page = (support.ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        for path in ("/", "/index.html"):
            STATE["calls"].clear()
            response = self.client.get(path, headers={"User-Agent": "tester", "Referer": "https://example.test/x?q=1"})
            self.assertEqual((response.status_code, response.headers["content-type"]), (200, "text/html; charset=utf-8"))
            self.assertEqual(response.text, page)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual([(call["kind"], call["agent"], call["referrer"], call["owner"]) for call in STATE["calls"]],
                             [("visit", "tester", "https://example.test/x?q=1", False)])
            self.assertNotIn("referrer-policy", response.headers)   # the app pages keep today's headers
            STATE["calls"].clear()
            head = self.client.head(path)
            self.assertEqual((head.status_code, head.headers["content-type"]), (200, "text/html; charset=utf-8"))
            self.assertEqual(head.headers["content-length"], str(len(page.encode("utf-8"))))
            self.assertEqual(head.content, b"")
            self.assertEqual(STATE["calls"], [])   # uptime probes are not visits

    def test_a_visit_from_the_owner_browser_is_tagged_and_a_referrer_holding_the_key_is_not_stored(self):
        self.client.cookies.set("fyd_owner", track.owner_token())
        self.client.get("/")
        self.assertTrue(STATE["calls"][-1]["owner"])
        self.client.cookies.set("fyd_owner", "wrong")
        self.client.get("/")
        self.assertFalse(STATE["calls"][-1]["owner"])
        self.client.cookies.clear()
        self.client.get("/", headers={"Referer": f"http://testserver/whoami/{STATE['key']}"})
        self.assertEqual(STATE["calls"][-1]["referrer"], "")
        self.client.get("/", headers={"Referer": f"http://testserver/stats/{STATE['key']}?x=1"})
        self.assertEqual(STATE["calls"][-1]["referrer"], "")
        self.assertNotIn(STATE["key"], json.dumps(usage_rows()))

    def test_samples_and_sample_files(self):
        self.assertEqual(api.SAMPLE_META, legacy_server.SAMPLE_META)   # the copy cannot drift from the live server
        self.assertEqual(api.SAMPLE_ORDER, legacy_server.SAMPLE_ORDER)
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        expected = {"samples": [{"name": name, **legacy_server.SAMPLE_META[name]} for name in legacy_server.SAMPLE_META
                                if (legacy_server.SAMPLES / name).exists()], "order": legacy_server.SAMPLE_ORDER}
        self.assertEqual(response.json(), expected)
        self.assertEqual(response.text, json.dumps(expected))   # plain json.dumps, as the old server wrote it
        for name in legacy_server.SAMPLE_META:
            response = self.client.get("/api/sample", params={"name": name})
            self.assertEqual((response.status_code, response.headers["content-type"]), (200, "text/csv"), name)
            self.assertEqual(response.text, (legacy_server.SAMPLES / name).read_text(encoding="utf-8"))
        for bad in ("", "nope.csv", "../README.md", "..\\README.md", "liver_disease.csv/../../Dockerfile", "%00"):
            response = self.client.get("/api/sample", params={"name": bad})
            self.assertEqual((response.status_code, response.text), (404, "not found"), bad)
        self.assertEqual(self.client.get("/api/sample").status_code, 404)
        self.assertEqual(STATE["calls"], [])   # nothing is logged for sample downloads

    def test_the_stats_key_is_accepted_three_ways_and_without_a_valid_key_the_answer_is_404(self):
        key = STATE["key"]
        STATE["saved"][2]("visit", "", "1.2.3.4", "agent", "", is_owner=0)   # the unwrapped function: not a call to count
        for response in (self.client.get("/stats", headers={"X-Stats-Key": key}), self.client.get("/stats", params={"key": key}),
                         self.client.get(f"/stats/{key}"), self.client.get(f"/stats/{key}/"),
                         self.client.get("/stats", headers={"x-stats-key": key})):
            self.assertEqual((response.status_code, response.headers["content-type"]), (200, "text/html; charset=utf-8"))
            self.assertIn("private stats", response.text)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.headers["referrer-policy"], "no-referrer")
            self.assertNotIn(key, response.text)
        self.assertEqual(self.client.head("/stats", headers={"X-Stats-Key": key}).status_code, 200)
        for response in (self.client.get("/stats"), self.client.get("/stats/"), self.client.get("/stats", params={"key": "wrong"}),
                         self.client.get("/stats/wrong"), self.client.get("/stats", headers={"X-Stats-Key": "wrong"}),
                         self.client.get("/stats", params={"key": key + "x"}), self.client.get("/stats", params={"key": key[:-1]}),
                         self.client.get("/stats", params={"key": ""}), self.client.get(f"/stats/{key}/extra")):
            self.assertEqual((response.status_code, response.text, response.headers["content-type"]),
                             (404, "not found", "text/plain"))
        with mock.patch.object(track, "STATS_KEY", ""):   # no key configured: no key opens the page
            for response in (self.client.get("/stats"), self.client.get("/stats", params={"key": ""}),
                             self.client.get("/stats/"), self.client.get("/stats", headers={"X-Stats-Key": ""}),
                             self.client.get("/whoami/"), self.client.get(f"/stats/{key}")):
                self.assertEqual((response.status_code, response.text), (404, "not found"))
        self.assertEqual(STATE["calls"], [])

    def test_whoami_sets_the_owner_cookie_only_for_the_key_and_never_echoes_it(self):
        key = STATE["key"]
        response = self.client.get(f"/whoami/{key}")
        self.assertEqual((response.status_code, response.headers["content-type"]), (200, "text/html; charset=utf-8"))
        cookie = response.headers["set-cookie"]
        self.assertTrue(cookie.startswith(f"fyd_owner={track.owner_token()};"), cookie)
        self.assertIn("Max-Age=31536000", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertNotIn(key, response.text + cookie)
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        self.assertEqual(response.headers["cache-control"], "no-store")
        for other in ("wrong", "", key + "1"):
            denied = self.client.get(f"/whoami/{other}")
            self.assertEqual((denied.status_code, denied.text), (404, "not found"))
            self.assertNotIn("set-cookie", denied.headers)
        self.client.cookies.clear()

    def test_the_client_ip_for_the_daily_hash_ignores_x_forwarded_for_unless_told_to_trust_it(self):
        spoof = {"X-Forwarded-For": "6.6.6.6, 7.7.7.7"}
        with mock.patch.dict("os.environ", {}):
            for name in ("FLY_APP_NAME", "FYD_TRUST_FLY_CLIENT_IP", "FYD_TRUST_X_FORWARDED_FOR"):
                os.environ.pop(name, None)
            self.client.get("/", headers=spoof)
            self.assertEqual(STATE["calls"][-1]["ip"], "testclient")
            with mock.patch.dict("os.environ", {"FYD_TRUST_X_FORWARDED_FOR": "1"}):
                self.client.get("/", headers=spoof)
                self.assertEqual(STATE["calls"][-1]["ip"], "6.6.6.6")
            with mock.patch.dict("os.environ", {"FYD_TRUST_X_FORWARDED_FOR": "0"}):
                self.client.get("/", headers=spoof)
                self.assertEqual(STATE["calls"][-1]["ip"], "testclient")

    def test_fly_client_ip_is_trusted_only_on_fly_or_when_told_to(self):
        fly = {"Fly-Client-IP": "203.0.113.9"}
        spoof = {"X-Forwarded-For": "6.6.6.6", **fly}
        with mock.patch.dict("os.environ", {}):
            for name in ("FLY_APP_NAME", "FYD_TRUST_FLY_CLIENT_IP", "FYD_TRUST_X_FORWARDED_FOR"):
                os.environ.pop(name, None)
            self.assertFalse(api.trust_fly_client_ip())
            self.client.get("/", headers=fly)       # not behind Fly: any caller could have sent this header
            self.assertEqual(STATE["calls"][-1]["ip"], "testclient")
            self.client.get("/", headers=spoof)
            self.assertEqual(STATE["calls"][-1]["ip"], "testclient")
            with mock.patch.dict("os.environ", {"FLY_APP_NAME": "some-app"}):   # Fly sets this in its machines
                self.assertTrue(api.trust_fly_client_ip())
                self.client.get("/", headers=fly)
                self.assertEqual(STATE["calls"][-1]["ip"], "203.0.113.9")
                self.client.get("/", headers=spoof)  # X-Forwarded-For is still ignored
                self.assertEqual(STATE["calls"][-1]["ip"], "203.0.113.9")
                self.client.get("/")                 # no header: the socket peer
                self.assertEqual(STATE["calls"][-1]["ip"], "testclient")
            with mock.patch.dict("os.environ", {"FLY_APP_NAME": "   "}):        # empty is not set
                self.assertFalse(api.trust_fly_client_ip())
            with mock.patch.dict("os.environ", {"FYD_TRUST_FLY_CLIENT_IP": "1"}):
                self.client.get("/", headers=fly)
                self.assertEqual(STATE["calls"][-1]["ip"], "203.0.113.9")
            with mock.patch.dict("os.environ", {"FYD_TRUST_FLY_CLIENT_IP": "0"}):
                self.client.get("/", headers=fly)
                self.assertEqual(STATE["calls"][-1]["ip"], "testclient")
            with mock.patch.dict("os.environ", {"FLY_APP_NAME": "some-app", "FYD_TRUST_X_FORWARDED_FOR": "1"}):
                self.client.get("/", headers={"X-Forwarded-For": "6.6.6.6"})   # Fly header absent: forwarded for
                self.assertEqual(STATE["calls"][-1]["ip"], "6.6.6.6")

    def test_health_routes_are_the_same_as_before(self):
        for path in ("/healthz", "/health"):
            response = self.client.get(path)
            self.assertEqual((response.status_code, response.text, response.headers["content-type"]),
                             (200, "ok", "text/plain"))
        self.assertEqual(STATE["calls"], [])

    @classmethod
    def tearDownClass(cls):
        cls.secrets.append(STATE["key"])   # the owner key must not be in any log line of this class
        super().tearDownClass()


# ============================================================================================ busy lane

class LegacyBusy(LegacyCase):
    CONFIG = {"max_waiting_heavy": 0}

    def test_a_busy_heavy_lane_gives_429_in_the_legacy_shape_and_the_light_lane_still_works(self):
        dataset = self.make_dataset()
        busy = self.submit(dataset, models=["erf", "csrboost", "balanced_rf", "smote_rf"])
        started = time.monotonic()
        for path, body in (("/api/analyze", {"csv": CSV_TEXT, "target": TARGET, "folds": 1}),
                           ("/api/score", {"train_csv": CSV_TEXT, "target": TARGET, "score_csv": SMALL_SCORE_CSV})):
            response = self.post(path, body)
            error = self.assertLegacyError(response, 429)
            self.assertIn("busy", error.lower())
            self.assertGreaterEqual(int(response.headers["retry-after"]), 1)
        self.assertLess(time.monotonic() - started, 2.0)
        light = self.post("/api/profile", {"csv": CSV_TEXT})
        self.assertEqual(light.status_code, 200)
        self.assertEqual(self.cancel(busy).json()["state"], "cancelled")
        wait_for(lambda: not processes_with(self.marker), 20, 1.0, "every child process to be gone")
        ok = self.post("/api/analyze", {"csv": CSV_TEXT, "target": TARGET, "folds": 1})
        self.assertEqual(ok.status_code, 200)   # the slot is free again

    def cancel(self, job):
        return self.client.delete(f"/api/v1/jobs/{job['job_id']}", headers=auth(job["capability"]))


# ============================================================================================ F1 and F2 over HTTP

def large_csv(rows: int) -> str:
    """A synthetic table with a rare class: about 65 bytes a row, and the legacy analysis result is about 2.6 times
    the input (the engine returns the balanced training table)."""
    import random
    rng = random.Random(1)
    lines = ["f0,f1,f2,f3,f4,f5,f6,f7,cat,label"]
    for _ in range(rows):
        positive = rng.random() < 0.08
        values = [f"{rng.gauss(1.0 if positive else 0.0, 1):.4f}" for _ in range(8)]
        lines.append(",".join(values) + f",{rng.choice('abcd')},{'yes' if positive else 'no'}")
    return "\n".join(lines) + "\n"


class LegacyLargeResult(LegacyCase):
    """The old server relayed any size; the 8 MB cap of the v1 jobs must not apply to the legacy analysis."""

    def test_an_analysis_whose_result_is_over_the_v1_cap_is_relayed(self):
        csv_text = large_csv(60_000)   # about 3.9 MB of CSV
        response = self.post("/api/analyze", {"csv": csv_text, "target": "label", "method": "smote", "folds": 1,
                                              "src": "upload"})
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertGreater(len(response.content), 8 * 1024 * 1024)   # more than the v1 cap, less than the legacy cap
        self.assertLess(len(response.content), 32 * 1024 * 1024)
        result = response.json()
        self.assertIn("balanced_train_csv", result)
        self.assertEqual(self.coordinator.stats()["jobs"], 0)
        self.assertLess(self.coordinator.stats()["stored_bytes"], 10_000)


class LegacySmallCap(LegacyCase):
    CONFIG = {"max_legacy_result_bytes": 20 * 1024}

    def test_a_result_over_the_cap_is_a_400_in_the_legacy_shape_with_a_clear_text(self):
        body = {"csv": CSV_TEXT, "target": TARGET, "method": "smote", "folds": 1}
        error = self.assertLegacyError(self.post("/api/analyze", body), 400)
        self.assertEqual(error, "The result is larger than the service allows. The limit is 20 KB. Try a smaller file.")
        self.assertEqual(self.coordinator.stats()["jobs"], 0)
        convert = self.post("/api/convert", {"b64": base64.b64encode(SAMPLE.read_bytes() * 2).decode(), "name": "x.csv"})
        self.assertLegacyError(convert, 400, error)   # the light lane has the same cap
        self.assertEqual(self.post("/api/profile", {"csv": CSV_TEXT}).status_code, 200)   # a small result still passes


class LegacyMemoryCeiling(LegacyCase):
    CONFIG = {"child_memory_mb": 60}   # the engine imports more than that, so every legacy call goes over it

    def test_a_legacy_call_over_the_ceiling_is_a_400_in_the_legacy_shape_and_nothing_stays_behind(self):
        for path, body in (("/api/profile", {"csv": CSV_TEXT}),
                           ("/api/analyze", {"csv": CSV_TEXT, "target": TARGET, "folds": 1})):
            error = self.assertLegacyError(self.post(path, body), 400)
            self.assertEqual(error, "This request needs more memory than the service allows for one job. "
                                    "Try a smaller file.")
        wait_for(lambda: not processes_with(self.marker), 20, 0.5, "every child process to be gone")
        self.assertEqual(self.client.get("/healthz").text, "ok")
        self.assertEqual(self.coordinator.stats()["jobs"], 0)
        self.assertEqual(self.coordinator.stats()["heavy_running"], 0)


# ============================================================================================ disconnect and cancel

class LegacyCancelOnDisconnect(LegacyCase):
    """The heavy legacy routes are async and cancel their job when the client goes away."""

    PAYLOAD = json.dumps(LONG_ANALYZE).encode("utf-8")

    def assertCleanAfter(self):
        stats = self.coordinator.stats()
        self.assertEqual((stats["heavy_running"], stats["heavy_waiting"]), (0, 0), stats)
        self.assertEqual(stats["jobs"], 0, stats)   # the cancelled job's record was collected too
        wait_for(lambda: not processes_with(self.marker), 20, 1.0, "every child process to be gone")

    async def drive(self, how: str) -> dict:
        disconnect = asyncio.Event()
        delivered = {"body": False}
        sent: list = []

        async def receive():
            if not delivered["body"]:
                delivered["body"] = True
                return {"type": "http.request", "body": self.PAYLOAD, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "path": "/api/analyze", "raw_path": b"/api/analyze", "query_string": b"", "scheme": "http",
                 "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"),
                             (b"content-length", str(len(self.PAYLOAD)).encode())],
                 "client": ("testclient", 1234), "server": ("testserver", 80)}
        task = asyncio.ensure_future(self.app(scope, receive, send))
        deadline = time.monotonic() + 60
        while self.coordinator.stats()["heavy_running"] != 1:
            self.assertLess(time.monotonic(), deadline, "the legacy job never started")
            self.assertFalse(task.done(), "the request ended before the job started")
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)   # the job is really running now
        self.assertFalse(task.done())
        began = time.monotonic()
        if how == "disconnect":
            disconnect.set()
        else:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return {"seconds": time.monotonic() - began, "sent": sent, "cancelled": task.cancelled()}

    def test_a_client_disconnect_cancels_the_job_and_kills_its_process_tree(self):
        outcome = asyncio.run(self.drive("disconnect"))
        self.assertLess(outcome["seconds"], 30)
        self.assertCleanAfter()

    def test_cancelling_the_request_task_cancels_the_job_and_kills_its_process_tree(self):
        outcome = asyncio.run(self.drive("cancel"))
        self.assertTrue(outcome["cancelled"])
        self.assertLess(outcome["seconds"], 30)
        self.assertCleanAfter()

    def test_the_event_loop_is_not_blocked_while_a_legacy_job_runs(self):
        async def scenario():
            delays = []

            async def ticker():
                last = time.perf_counter()
                for _ in range(40):
                    await asyncio.sleep(0.05)
                    now = time.perf_counter()
                    delays.append(now - last - 0.05)
                    last = now
            ticking = asyncio.ensure_future(ticker())
            outcome = await self.drive("disconnect")
            await ticking
            return delays, outcome
        delays, _ = asyncio.run(scenario())
        self.assertLess(max(delays), 0.5, delays)   # a blocked loop would show one long gap
        self.assertCleanAfter()


class LegacyRealServer(unittest.TestCase):
    """A real client disconnect against a real Uvicorn process."""

    @classmethod
    def setUpClass(cls):
        cls.base = support.private_directory("fyd-compat-server-")
        cls.marker = cls.base.name
        cls.server = support.ServerProcess(cls.base, {"FYD_USAGE_DB": str(cls.base / "usage.db")})
        try:
            cls.server.wait_ready()
        except BaseException:
            cls.server.stop()
            support.remove_directory(cls.base)
            raise

    @classmethod
    def tearDownClass(cls):
        problems = []
        code = cls.server.stop()
        if not cls.server.stopped_cleanly(code):
            problems.append(f"the server did not shut down cleanly (exit code {code})")
        leftover = processes_with(cls.marker)
        for pid in leftover:
            support.kill_pid(pid)
        if leftover:
            problems.append(f"{len(leftover)} process(es) left")
        if cls.server.root.exists():
            problems.append("the storage root still exists")
        output = cls.server.output()
        if "Traceback" in output:
            problems.append("a traceback is in the server output")
        support.remove_directory(cls.base)
        if cls.base.exists():
            problems.append("the private directory could not be removed")
        if problems:
            raise AssertionError("; ".join(problems))

    def test_closing_the_connection_during_an_analysis_kills_the_children_and_frees_the_lane(self):
        payload = json.dumps(LONG_ANALYZE).encode("utf-8")
        request = (b"POST /api/analyze HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                   b"Connection: close\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        with socket.create_connection(("127.0.0.1", self.server.port), timeout=10) as sock:
            sock.sendall(request)
            wait_for(lambda: processes_with(self.marker), 60, 1.0, "the analysis child process to appear")
        # the socket is closed now: the server must notice, cancel the job and kill the tree
        wait_for(lambda: not processes_with(self.marker), 40, 1.0, "the child processes to be killed after the disconnect")
        with httpx.Client(base_url=self.server.url, timeout=120) as http:
            self.assertEqual(http.get("/healthz").status_code, 200)
            fast = http.post("/api/analyze", json={"csv": CSV_TEXT, "target": TARGET, "folds": 1})
            self.assertEqual(fast.status_code, 200, fast.text[:200])   # not 429: the heavy lane was freed
            profile = http.post("/api/profile", json={"csv": CSV_TEXT})
            self.assertEqual(profile.status_code, 200)
            self.assertEqual(profile.json()["suggested_target"], TARGET)
            self.assertEqual(http.get("/").status_code, 200)
            self.assertEqual(http.get("/stats/anything").text, "not found")   # no key configured in this process


if __name__ == "__main__":
    unittest.main()
