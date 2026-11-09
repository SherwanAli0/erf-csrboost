"""Behaviour tests for the /api/v1 interface (``backend/api.py``).

Most classes run the real app with a real coordinator in this process (Starlette's ``TestClient``, lifespan
included), real child processes and real training on a synthetic table of 300 rows. ``V1Process`` starts the real
server with ``python backend/api.py`` on a free port, because event-loop behaviour (health during a fit, a
chunked upload with no Content-Length, shutdown with a running job) is only visible there.

Every class captures the whole process log and fails if a capability appears in it, and fails if a child process
or a storage root is left behind. Private directories are removed at the end.

Run from the repository root::

    python -B -m unittest discover -s tests -p "test_api_v1.py" -v
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import threading
import time
import unittest

import api_support as support  # noqa: F401  (sets sys.path and warning filters before the imports below)
from api_support import ApiCase, LogCapture, auth, job_request, processes_with, strip_request_id, synthetic_csv, wait_for

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from jobs import Coordinator, JobConfig, states  # noqa: E402

LONG_MODELS = ["erf", "csrboost", "balanced_rf", "smote_rf"]   # several seconds on 300 rows
FAST_MODELS = ["logistic", "rf_weighted"]
HOSTILE_POSITIVE = "<script>alert(1)</script>"
HOSTILE_NEGATIVE = "=cmd|' /C calc'!A0"


# ============================================================================================ a real job, end to end

class V1RealJob(ApiCase):
    """One real two-model training job (about 300 rows) shared by the read-only checks, plus two visitors."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            cls.alice_dataset = cls._create(synthetic_csv(), "alice.csv")
            cls.bob_dataset = cls._create(synthetic_csv(), "bob.csv")
            response = cls.client.post("/api/v1/jobs", json=job_request(cls.alice_dataset, models=FAST_MODELS),
                                       headers=auth(cls.alice_dataset["capability"]))
            assert response.status_code == 202, response.text
            cls.alice_job = response.json()
            cls.secrets.append(cls.alice_job["capability"])
            cls.jobs.append(cls.alice_job)
            cls.final = cls._wait(cls.alice_job)
        except BaseException:
            super().tearDownClass()
            raise

    @classmethod
    def _create(cls, raw: bytes, name: str) -> dict:
        response = cls.client.post("/api/v1/datasets", content=raw,
                                   headers={"Content-Type": "text/csv", "X-Filename": name})
        assert response.status_code == 201, response.text
        body = response.json()
        cls.secrets.append(body["capability"])
        return body

    @classmethod
    def _wait(cls, job: dict) -> dict:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status = cls.client.get(f"/api/v1/jobs/{job['job_id']}", headers=auth(job["capability"])).json()
            if status["state"] not in ("queued", "running"):
                return status
            time.sleep(0.1)
        raise AssertionError("the real job did not finish")

    def test_the_dataset_response_holds_the_profile_limits_and_a_one_time_capability(self):
        dataset = self.alice_dataset
        self.assertEqual(set(dataset), {"dataset_id", "capability", "profile", "limits", "expires_at"})
        self.assertEqual(dataset["profile"]["schema"], "app-profile-1.0")
        self.assertEqual(dataset["profile"]["rows"], 300)
        self.assertEqual(dataset["profile"]["target"]["suggestion"]["column"], "label")
        self.assertEqual(dataset["limits"]["upload_limit_bytes"], 1_000_000)
        self.assertRegex(dataset["expires_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertNotIn(dataset["capability"], json.dumps(dataset["profile"]))
        self.assertNotIn(str(self.base), json.dumps(dataset))
        # the capability is not repeated by any later read of the dataset's jobs
        self.assertNotIn(dataset["capability"], json.dumps(self.final))

    def test_job_creation_response_and_status_of_a_finished_job(self):
        job = self.alice_job
        self.assertEqual(set(job), {"job_id", "capability", "state", "poll_url"})
        self.assertEqual(job["state"], "queued")
        self.assertEqual(job["poll_url"], f"/api/v1/jobs/{job['job_id']}")
        status = self.final
        self.assertEqual(status["state"], "succeeded", status)
        self.assertEqual(status["progress"]["stage"], "done")
        self.assertEqual([item["model"] for item in status["progress"]["models_completed"]], FAST_MODELS)
        self.assertEqual({item["status"] for item in status["models"]}, {"succeeded"})
        self.assertEqual(status["limits"]["per_model_timeout_seconds"], 60)
        self.assertIsNone(status["queue_position"])
        self.assertIsNotNone(status["expires_at"])
        self.assertEqual(self.final["progress"]["models_selected"], FAST_MODELS)

    def test_the_result_is_the_package_report_and_never_a_percentage(self):
        response = self.client.get(f"/api/v1/jobs/{self.alice_job['job_id']}/result",
                                   headers=auth(self.alice_job["capability"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.headers["cache-control"], "no-store")
        report = response.json()
        self.assertEqual(report["schema_version"], "training-1.1")
        self.assertEqual({name: row["status"] for name, row in report["outcomes"].items()},
                         {"logistic": "success", "rf_weighted": "success"})
        self.assertEqual(report["label_mapping"]["positive_label"], "fraud")
        self.assertNotIn(str(self.base), response.text)
        self.assertNoSecretIn(response.text)
        for forbidden in ("percent", "eta", "remaining"):
            self.assertNotIn(forbidden, json.dumps(self.final).lower())

    def test_every_export_has_the_right_type_disposition_and_no_store(self):
        base = f"/api/v1/jobs/{self.alice_job['job_id']}/exports/"
        headers = auth(self.alice_job["capability"])
        expected = {"report_json": ("application/json", "fyd-report.json"),
                    "report_html": ("text/html; charset=utf-8", "fyd-report.html"),
                    "config_json": ("application/json", "fyd-config.json")}
        bodies = {}
        for kind, (content_type, filename) in expected.items():
            response = self.client.get(base + kind, headers=headers)
            self.assertEqual(response.status_code, 200, kind)
            self.assertEqual(response.headers["content-type"], content_type, kind)
            self.assertEqual(response.headers["content-disposition"], f'attachment; filename="{filename}"', kind)
            self.assertEqual(response.headers["cache-control"], "no-store", kind)
            self.assertEqual(response.headers["x-content-type-options"], "nosniff", kind)
            bodies[kind] = response
        result = self.client.get(f"/api/v1/jobs/{self.alice_job['job_id']}/result", headers=headers).json()
        self.assertEqual(bodies["report_json"].json(), result)
        config = bodies["config_json"].json()
        self.assertEqual(config["request"]["models"], FAST_MODELS)
        self.assertEqual(config["input_sha256"], result["input_sha256"])
        self.assertIn("<html", bodies["report_html"].text.lower())
        self.assertIn("default-src 'none'", bodies["report_html"].headers["content-security-policy"])
        for kind in expected:
            self.assertNoSecretIn(bodies[kind].text)
        # the allowlist: anything else is a typed 404, whatever the capability
        for bad in ("report_csv", "REPORT_JSON", "..%2F..%2Fresult", "report_json.", "config"):
            response = self.client.get(base + bad, headers=headers)
            self.assertEqual(response.status_code, 404, bad)
            self.assertEqual(response.json()["error"]["code"], "EXPORT_UNKNOWN" if "%" not in bad else "NOT_FOUND")

    def test_a_second_visitor_gets_the_same_404_for_every_route_and_changes_nothing(self):
        alice, bob = self.alice_job, self.bob_dataset
        job_id, dataset_id = alice["job_id"], self.alice_dataset["dataset_id"]
        unknown_job, unknown_dataset = "job_" + "A" * 22, "ds_" + "B" * 22
        probes = {
            "status": lambda jid, token: self.client.get(f"/api/v1/jobs/{jid}", headers=auth(token)),
            "cancel": lambda jid, token: self.client.delete(f"/api/v1/jobs/{jid}", headers=auth(token)),
            "result": lambda jid, token: self.client.get(f"/api/v1/jobs/{jid}/result", headers=auth(token)),
            "report_json": lambda jid, token: self.client.get(f"/api/v1/jobs/{jid}/exports/report_json",
                                                              headers=auth(token)),
            "report_html": lambda jid, token: self.client.get(f"/api/v1/jobs/{jid}/exports/report_html",
                                                              headers=auth(token)),
            "config_json": lambda jid, token: self.client.get(f"/api/v1/jobs/{jid}/exports/config_json",
                                                              headers=auth(token)),
        }
        wrong_tokens = {"bob dataset capability": bob["capability"], "alice dataset capability (wrong kind)":
                        self.alice_dataset["capability"], "garbage": "x" * 43, "empty": ""}
        for name, probe in probes.items():
            reference = probe(unknown_job, bob["capability"])
            self.assertEqual(reference.status_code, 404, name)
            self.assertEqual(reference.json()["error"]["code"], "NOT_FOUND")
            for label, token in wrong_tokens.items():
                response = probe(job_id, token)
                self.assertEqual(response.status_code, 404, (name, label))
                self.assertEqual(strip_request_id(response), strip_request_id(reference), (name, label))
            response = probe(job_id, "")
            no_header = self.client.get(f"/api/v1/jobs/{job_id}") if name == "status" else response
            self.assertEqual(no_header.status_code, 404)
        # a header that is not a bearer header counts as no token
        odd = self.client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Basic {alice['capability']}"})
        self.assertEqual(odd.status_code, 404)
        self.assertNotIn(alice["capability"], odd.text)
        # deleting the dataset
        reference = self.client.delete(f"/api/v1/datasets/{unknown_dataset}", headers=auth(bob["capability"]))
        for label, token in {"bob": bob["capability"], "alice job capability": alice["capability"], "none": ""}.items():
            response = self.client.delete(f"/api/v1/datasets/{dataset_id}", headers=auth(token))
            self.assertEqual(response.status_code, 404, label)
            self.assertEqual(strip_request_id(response), strip_request_id(reference), label)
        # a job on someone else's dataset: same refusal as an unknown dataset
        response = self.client.post("/api/v1/jobs", json=job_request(self.alice_dataset), headers=auth(bob["capability"]))
        unknown = self.client.post("/api/v1/jobs", json={**job_request(self.alice_dataset), "dataset_id": unknown_dataset},
                                   headers=auth(bob["capability"]))
        self.assertEqual((response.status_code, unknown.status_code), (404, 404))
        self.assertEqual(strip_request_id(response), strip_request_id(unknown))
        # nothing happened to Alice's job or dataset
        self.assertEqual(self.status(alice)["state"], "succeeded")
        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}/result", headers=auth(alice["capability"])).status_code, 200)
        for response in (reference,):
            self.assertNoSecretIn(response.text)
        # request ids differ per request, so a body comparison is not trivially equal
        first = self.client.get(f"/api/v1/jobs/{job_id}", headers=auth(bob["capability"]))
        second = self.client.get(f"/api/v1/jobs/{job_id}", headers=auth(bob["capability"]))
        self.assertNotEqual(first.headers["x-request-id"], second.headers["x-request-id"])
        self.assertEqual(first.json()["error"]["request_id"], first.headers["x-request-id"])

    def test_deleting_a_dataset_removes_it_and_a_repeat_is_a_404(self):
        dataset = self.make_dataset(name="gone.csv")
        before = self.coordinator.stats()["datasets"]
        wrong = self.client.delete(f"/api/v1/datasets/{dataset['dataset_id']}", headers=auth("nope"))
        self.assertEqual(wrong.status_code, 404)
        self.assertEqual(self.coordinator.stats()["datasets"], before)
        response = self.client.delete(f"/api/v1/datasets/{dataset['dataset_id']}", headers=auth(dataset["capability"]))
        self.assertEqual((response.status_code, response.json()), (200, {"deleted": True}))
        self.assertEqual(self.coordinator.stats()["datasets"], before - 1)
        again = self.client.delete(f"/api/v1/datasets/{dataset['dataset_id']}", headers=auth(dataset["capability"]))
        self.assertEqual(again.status_code, 404)
        late = self.client.post("/api/v1/jobs", json=job_request(dataset), headers=auth(dataset["capability"]))
        self.assertEqual(late.status_code, 404)

    def test_hostile_labels_are_escaped_in_html_and_documented_for_json(self):
        columns = support.mixed_table(n=300, positives=40, labels=(HOSTILE_POSITIVE, HOSTILE_NEGATIVE))
        raw_response = self.client.post("/api/v1/datasets", content=support.table_to_csv(columns),
                                        headers={"Content-Type": "text/csv"})
        self.assertEqual(raw_response.status_code, 201)
        self.assertNotIn(b"<", raw_response.content)   # the profile response is escaped as well
        self.assertIn(b"\u003cscript\u003e", raw_response.content)
        dataset = raw_response.json()
        self.assertEqual(dataset["profile"]["classes"]["suggested_positive"]["label"], HOSTILE_POSITIVE)
        self.remember(dataset["capability"])
        self.addCleanup(lambda: self.client.delete(f"/api/v1/datasets/{dataset['dataset_id']}",
                                                   headers=auth(dataset["capability"])))
        job = self.submit(dataset, positive_label=HOSTILE_POSITIVE, negative_label=HOSTILE_NEGATIVE,
                          models=["logistic"])
        status = self.wait_state(job, ("succeeded", "partial", "failed"))
        self.assertEqual(status["state"], "succeeded", status)
        headers = auth(job["capability"])
        html = self.client.get(f"/api/v1/jobs/{job['job_id']}/exports/report_html", headers=headers).text
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)   # shown as text
        self.assertNotIn(HOSTILE_NEGATIVE, html)                         # the apostrophes are escaped too
        self.assertIn("=cmd|&#x27; /C calc&#x27;!A0", html)
        # JSON: the label survives as a string value, but the bytes hold no markup characters
        for url in (f"/api/v1/jobs/{job['job_id']}/exports/report_json", f"/api/v1/jobs/{job['job_id']}/result",
                    f"/api/v1/jobs/{job['job_id']}/exports/config_json"):
            response = self.client.get(url, headers=headers)
            self.assertEqual(response.status_code, 200, url)
            self.assertNotIn(b"<script", response.content, url)
            self.assertNotIn(b"<", response.content, url)
            self.assertIn(b"\\u003cscript\\u003e", response.content, url)
        report = self.client.get(f"/api/v1/jobs/{job['job_id']}/exports/report_json", headers=headers).json()
        self.assertEqual(report["label_mapping"]["positive_label"], HOSTILE_POSITIVE)
        self.assertEqual(report["label_mapping"]["negative_label"], HOSTILE_NEGATIVE)
        config = self.client.get(f"/api/v1/jobs/{job['job_id']}/exports/config_json", headers=headers).json()
        self.assertEqual(config["request"]["positive_label"], HOSTILE_POSITIVE)


# ============================================================================================ typed errors, guards

class V1Errors(ApiCase):
    def error(self, response, status: int, code: str) -> dict:
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        self.assertEqual(set(body), {"error"})
        self.assertEqual(set(body["error"]), {"code", "message", "details", "request_id"})
        self.assertEqual(body["error"]["code"], code, body)
        self.assertEqual(body["error"]["request_id"], response.headers["x-request-id"])
        self.assertEqual(response.headers["content-type"], "application/json")
        for forbidden in ("Traceback", 'File "', str(self.base), "site-packages"):
            self.assertNotIn(forbidden, response.text)
        self.assertNoSecretIn(response.text)
        return body["error"]

    def test_bad_json_is_a_400_and_a_non_object_or_unknown_field_a_422(self):
        headers = {"Content-Type": "application/json", **auth("anything")}
        for raw in (b"{bad", b"", b"\xff\xfe", b'{"target": NaN}', b"[" * 5000):
            self.error(self.client.post("/api/v1/jobs", content=raw, headers=headers), 400, "INVALID_JSON")
        error = self.error(self.client.post("/api/v1/jobs", json=[1, 2], headers=auth("anything")), 422, "REQUEST_INVALID")
        self.assertTrue(error["details"]["errors"])
        body = {"dataset_id": "ds_x", "target": "label", "positive_label": "fraud", "negative_label": "ok",
                "surprise": 1, "costs": {"false_positive_cost": 1, "false_negative_cost": 5, "extra": 2}}
        error = self.error(self.client.post("/api/v1/jobs", json=body, headers=auth("anything")), 422, "REQUEST_INVALID")
        self.assertEqual(sorted(error["details"]["unknown_fields"]), ["costs.extra", "surprise"])
        for bad in ({"positive_label": True}, {"positive_label": None}, {"positive_label": ["a"]},
                    {"target": ""}, {"seed": 1.5}, {"seed": True}, {"models": []}, {"models": [1]},
                    {"one_vs_rest": "yes"}, {"review_fractions": ["a"]}, {"positive_label": float("inf")}):
            payload = {"dataset_id": "ds_x", "target": "label", "positive_label": "fraud", "negative_label": "ok", **bad}
            raw = json.dumps(payload).replace("Infinity", "1e999")
            response = self.client.post("/api/v1/jobs", content=raw, headers=headers)
            self.assertIn(response.status_code, (400, 422), bad)
            self.assertNoSecretIn(response.text)
        # a label of the wrong kind does not leak its value into the answer
        response = self.client.post("/api/v1/jobs", json={"dataset_id": "ds_x", "target": "t", "positive_label": {"v": "s3cret-marker"}},
                                    headers=auth("anything"))
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("s3cret-marker", response.text)

    def test_labels_may_be_text_or_numbers(self):
        dataset = self.make_dataset()
        for label in ("fraud", 1, 1.5):
            response = self.client.post("/api/v1/jobs", json={"dataset_id": "ds_unknown", "target": "label",
                                                              "positive_label": label, "negative_label": "ok"},
                                        headers=auth(dataset["capability"]))
            self.error(response, 404, "NOT_FOUND")   # the body was accepted; the unknown dataset is the only refusal

    def test_the_wrong_content_type_is_a_415_on_both_posting_routes(self):
        dataset_headers = {"Content-Type": "application/json"}
        error = self.error(self.client.post("/api/v1/datasets", content=b"a,b\n1,2\n", headers=dataset_headers), 415,
                           "UNSUPPORTED_MEDIA_TYPE")
        self.assertIn("text/csv", error["details"]["allowed"])
        self.error(self.client.post("/api/v1/datasets", content=b"a,b\n1,2\n", headers={"Content-Type": "text/plain"}),
                   415, "UNSUPPORTED_MEDIA_TYPE")
        self.error(self.client.post("/api/v1/datasets", content=b"a,b\n1,2\n"), 415, "UNSUPPORTED_MEDIA_TYPE")
        for content_type in ("text/plain", "application/x-www-form-urlencoded", "text/csv"):
            self.error(self.client.post("/api/v1/jobs", content=b"{}", headers={"Content-Type": content_type}), 415,
                       "UNSUPPORTED_MEDIA_TYPE")
        # parameters on an accepted media type are fine
        ok = self.client.post("/api/v1/datasets", content=synthetic_csv(), headers={"Content-Type": "text/csv; charset=utf-8"})
        self.assertEqual(ok.status_code, 201)
        self.remember(ok.json()["capability"])
        self.client.delete(f"/api/v1/datasets/{ok.json()['dataset_id']}", headers=auth(ok.json()["capability"]))

    def test_an_oversize_body_is_a_413_declared_streamed_or_in_a_job_request(self):
        limit = 1_000_000
        before = self.coordinator.stats()
        error = self.error(self.client.post("/api/v1/datasets", content=b"a" * (limit + 1),
                                            headers={"Content-Type": "text/csv"}), 413, "REQUEST_TOO_LARGE")
        self.assertEqual(error["details"]["limit_bytes"], limit)

        def chunks():
            for _ in range(60):   # 3 MB with no Content-Length
                yield b"x" * 50_000
        request = self.client.build_request("POST", "/api/v1/datasets", content=chunks(),
                                            headers={"Content-Type": "application/octet-stream"})
        self.assertNotIn("content-length", request.headers)
        self.assertEqual(request.headers.get("transfer-encoding"), "chunked")
        self.error(self.client.send(request), 413, "REQUEST_TOO_LARGE")
        self.error(self.client.post("/api/v1/jobs", content=b" " * (128 * 1024 + 1),
                                    headers={"Content-Type": "application/json"}), 413, "REQUEST_TOO_LARGE")
        self.error(self.client.post("/api/v1/jobs", content=b"{}" * 70000, headers={"Content-Type": "application/json"}),
                   413, "REQUEST_TOO_LARGE")
        after = self.coordinator.stats()
        self.assertEqual((after["datasets"], after["jobs"], after["light_waiting"]),
                         (before["datasets"], before["jobs"], 0))
        # a body of exactly the limit is accepted by the size check (the content then fails to profile)
        response = self.client.post("/api/v1/datasets", content=b"a" * limit, headers={"Content-Type": "text/csv"})
        self.assertNotEqual(response.status_code, 413)

    def test_a_malformed_content_length_is_a_400(self):
        response = self.client.post("/api/v1/datasets", content=b"a,b\n", headers={"Content-Type": "text/csv",
                                                                                  "Content-Length": "abc"})
        self.error(response, 400, "INVALID_CONTENT_LENGTH")

    def test_a_cross_origin_state_change_is_a_403_and_a_same_origin_one_is_not(self):
        headers = {"Content-Type": "text/csv"}
        error = self.error(self.client.post("/api/v1/datasets", content=synthetic_csv(),
                                            headers={**headers, "Origin": "https://evil.example"}), 403, "ORIGIN_NOT_ALLOWED")
        self.assertEqual(self.coordinator.stats()["datasets"], 0)
        self.error(self.client.post("/api/v1/datasets", content=b"x", headers={**headers, "Origin": "null"}), 403,
                   "ORIGIN_NOT_ALLOWED")
        self.error(self.client.delete("/api/v1/jobs/job_x", headers={"Origin": "http://other.test", **auth("t")}), 403,
                   "ORIGIN_NOT_ALLOWED")
        self.error(self.client.post("/api/v1/jobs", json={}, headers={"Origin": "http://testserver:9999"}), 403,
                   "ORIGIN_NOT_ALLOWED")
        same = self.client.delete("/api/v1/jobs/job_x", headers={"Origin": "http://testserver", **auth("t")})
        self.error(same, 404, "NOT_FOUND")   # the guard let it through; the unknown id is the answer
        self.assertTrue(api.origin_allowed(api.Headers({"origin": "https://app.example:443", "host": "app.example"})))
        self.assertTrue(api.origin_allowed(api.Headers({"origin": "http://app.example:80", "host": "APP.example"})))
        self.assertFalse(api.origin_allowed(api.Headers({"origin": "https://app.example", "host": "app.example:8443"})))
        # reads are not state changing, and no CORS header is ever sent
        response = self.client.get("/api/v1/version", headers={"Origin": "https://evil.example"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse([name for name in response.headers if name.lower().startswith("access-control-")])
        preflight = self.client.options("/api/v1/jobs", headers={"Origin": "https://evil.example",
                                                                 "Access-Control-Request-Method": "POST"})
        self.assertNotIn("access-control-allow-origin", preflight.headers)

    def test_unknown_routes_and_methods_are_typed_and_every_response_has_the_security_headers(self):
        self.error(self.client.get("/api/v1/nothing"), 404, "NOT_FOUND")
        response = self.client.put("/api/v1/jobs", content=b"{}")
        self.error(response, 405, "METHOD_NOT_ALLOWED")
        self.assertIn("POST", response.headers.get("allow", ""))
        self.error(self.client.get("/api/v1/jobs/"), 404, "NOT_FOUND")
        for response in (self.client.get("/healthz"), self.client.get("/api/v1/version"), self.client.get("/api/v1/samples"),
                         self.client.get("/api/v1/jobs/job_x"), self.client.post("/api/v1/jobs", content=b"{"),
                         self.client.get("/api/v1/nothing")):
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
            self.assertEqual(response.headers["x-frame-options"], "DENY")
            self.assertEqual(response.headers["referrer-policy"], "no-referrer")
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertRegex(response.headers["x-request-id"], r"^[0-9a-f]{16}$")

    def test_health_version_and_samples(self):
        for method in ("get", "head"):
            for path in ("/healthz", "/health"):
                response = getattr(self.client, method)(path)
                self.assertEqual((response.status_code, response.headers["content-type"]), (200, "text/plain"))
        self.assertEqual(self.client.get("/healthz").text, "ok")
        self.assertEqual(self.client.head("/healthz").content, b"")
        version = self.client.get("/api/v1/version").json()
        self.assertEqual(set(version), {"app_version", "package_version", "report_schemas", "release_id"})
        self.assertEqual(version["report_schemas"]["training"], "training-1.1")
        self.assertEqual(version["release_id"], "unknown")
        text = json.dumps(version)
        for forbidden in (str(self.base), str(support.ROOT), os.path.expanduser("~"), "\\\\", "C:\\", "/home/", "token", "key"):
            self.assertNotIn(forbidden, text)
        samples = self.client.get("/api/v1/samples").json()
        self.assertEqual({item["name"] for item in samples["samples"]}, set(api.SAMPLE_META))
        self.assertEqual(samples["order"], api.SAMPLE_ORDER)
        legacy = self.client.get("/api/samples").json()
        self.assertEqual(samples, legacy)

    def test_the_release_id_comes_from_the_environment_and_only_if_it_looks_like_an_id(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"FYD_RELEASE_ID": "abc-123.4"}):
            self.assertEqual(api._release_id(), "abc-123.4")
        for bad in ("C:\\secret\\path", "has space", "x" * 65, "/etc/passwd"):
            with mock.patch.dict(os.environ, {"FYD_RELEASE_ID": bad}):
                self.assertEqual(api._release_id(), "unknown")

    def test_an_unexpected_exception_is_a_typed_500_with_no_text_and_no_traceback_in_the_log(self):
        def boom(*args):
            raise RuntimeError("secret-detail C:\\private\\place token=abc123")
        from unittest import mock
        with mock.patch.object(self.coordinator, "create_dataset", boom):
            response = self.client.post("/api/v1/datasets", content=b"a,b\n1,2\n", headers={"Content-Type": "text/csv"})
        error = self.error(response, 500, "INTERNAL_ERROR")
        self.assertNotIn("secret-detail", response.text)
        lines = [line for line in self.log.lines if "unhandled error" in line]
        self.assertTrue(lines)
        self.assertNotIn("secret-detail", self.log.text)
        self.assertNotIn("Traceback", self.log.text)
        self.assertIn("class=RuntimeError", lines[-1])
        self.assertIn(error["request_id"], lines[-1])

    def test_a_client_that_disconnects_while_uploading_is_logged_at_info_and_not_as_an_error(self):
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/api/v1/datasets", "raw_path": b"/api/v1/datasets", "query_string": b"", "root_path": "",
                 "headers": [(b"host", b"testserver"), (b"content-type", b"text/csv")],
                 "client": ("127.0.0.1", 5000), "server": ("testserver", 80)}
        sent = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        before = len(self.log.lines)
        asyncio.run(self.app(scope, receive, send))
        lines = self.log.lines[before:]
        self.assertTrue(any(line.startswith("fyd.api INFO client disconnected request=") for line in lines), lines)
        self.assertFalse([line for line in lines if "unhandled error" in line or " ERROR " in line], lines)
        self.assertEqual(self.coordinator.stats()["datasets"], 0)   # nothing was stored for it
        # a real fault is still an error line
        def boom(*args):
            raise RuntimeError("detail")
        from unittest import mock
        with mock.patch.object(self.coordinator, "create_dataset", boom):
            self.client.post("/api/v1/datasets", content=b"a,b\n1,2\n", headers={"Content-Type": "text/csv"})
        self.assertTrue(any(" ERROR " in line and "unhandled error class=RuntimeError" in line
                            for line in self.log.lines[before:]))

    def test_the_coordinator_is_only_started_by_the_lifespan_and_stopped_by_it(self):
        self.assertTrue(self.coordinator.stats()["started"])
        base = support.private_directory()
        try:
            coordinator = Coordinator(JobConfig(temp_root=str(base / "root")))
            app = api.create_app(coordinator=coordinator)
            self.assertFalse(coordinator.stats()["started"])   # creating the app starts nothing
            with TestClient(app) as client:
                self.assertTrue(coordinator.stats()["started"])
                self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertFalse(coordinator.stats()["started"])
            self.assertFalse(coordinator.store.root.exists())
        finally:
            support.remove_directory(base)

    def test_importing_the_module_starts_nothing_and_app_is_created_on_first_use(self):
        import subprocess
        import sys
        code = ("import sys, threading; sys.path.insert(0, %r); import api; "
                "assert 'app' not in vars(api); assert threading.active_count() == 1; "
                "assert 'pandas' not in sys.modules and 'engine' not in sys.modules and 'track' not in sys.modules; "
                "application = api.app; assert api.app is application; "
                "assert threading.active_count() == 1; print('lazy-ok')" % str(support.BACKEND))
        output = subprocess.run([sys.executable, "-B", "-W", "ignore", "-c", code], capture_output=True, text=True,
                                timeout=120, cwd=str(self.base))
        self.assertEqual(output.stdout.strip(), "lazy-ok", output.stderr[-500:])


# ============================================================================================ result codes per state

class V1ResultStates(ApiCase):
    CONFIG = {"finished_job_seconds": 5.0}

    def result(self, job, token=None):
        return self.client.get(f"/api/v1/jobs/{job['job_id']}/result", headers=auth(token or job["capability"]))

    def test_202_while_running_then_409_after_a_cancel(self):
        dataset = self.make_dataset()
        job = self.submit(dataset, models=LONG_MODELS)
        response = self.result(job)
        self.assertEqual(response.status_code, 202)
        self.assertIn(response.json()["state"], ("queued", "running"))
        self.assertEqual(response.headers["retry-after"], "2")
        self.wait_fitting(job)
        response = self.result(job)
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["progress"]["stage"], "fitting")
        self.assertTrue(body["progress"]["current_model"])
        self.assertEqual(self.client.get(f"/api/v1/jobs/{job['job_id']}/exports/report_json",
                                         headers=auth(job["capability"])).status_code, 409)
        self.assertEqual(self.cancel(job).json()["state"], "cancelled")
        response = self.result(job)
        self.assertEqual(response.status_code, 409)
        error = response.json()["error"]
        self.assertEqual(error["code"], "RESULT_NOT_AVAILABLE")
        self.assertEqual(error["details"]["state"], "cancelled")
        self.assertEqual(error["details"]["reason"]["code"], "CANCELLED")

    def test_409_for_a_failed_job_with_the_reason_and_200_then_410_for_a_job_that_expired(self):
        dataset = self.make_dataset()
        failed = self.submit(dataset, target="no_such_column", models=["logistic"])
        status = self.wait_state(failed, ("failed", "succeeded", "partial"))
        self.assertEqual(status["state"], "failed")
        response = self.result(failed)
        self.assertEqual(response.status_code, 409)
        error = response.json()["error"]
        self.assertEqual((error["code"], error["details"]["state"]), ("RESULT_NOT_AVAILABLE", "failed"))
        self.assertEqual(error["details"]["reason"]["code"], status["reason"]["code"])
        self.assertNotIn(str(self.base), response.text)
        # the same dataset can run again; this one succeeds, then ages out
        good = self.submit(dataset, models=["logistic"])
        self.assertEqual(self.wait_state(good, ("succeeded", "partial", "failed"))["state"], "succeeded")
        self.assertEqual(self.result(good).status_code, 200)
        wait_for(lambda: self.status(good)["state"] == "expired", 40, 0.2, "the finished job to expire")
        response = self.result(good)
        self.assertEqual(response.status_code, 410)
        self.assertEqual(response.json()["error"]["code"], "JOB_EXPIRED")
        export = self.client.get(f"/api/v1/jobs/{good['job_id']}/exports/report_html", headers=auth(good["capability"]))
        self.assertEqual((export.status_code, export.json()["error"]["code"]), (410, "JOB_EXPIRED"))
        # a stranger still gets the plain 404 for the expired job, not a 410
        stranger = self.result(good, token=dataset["capability"])
        self.assertEqual(stranger.status_code, 404)


# ============================================================================================ overload and cancel

class V1Overload(ApiCase):
    CONFIG = {"max_waiting_heavy": 1}

    def assertNoChildren(self) -> None:
        wait_for(lambda: not processes_with(self.marker), 20, 1.0, "every child process to be gone")

    def test_a_full_queue_is_a_prompt_429_and_cancelling_queued_then_running_jobs_leaves_no_process(self):
        first, second, third = self.make_dataset(), self.make_dataset(), self.make_dataset()
        running = self.submit(first, models=LONG_MODELS)
        queued = self.submit(second, models=LONG_MODELS)
        started = time.monotonic()
        response = self.client.post("/api/v1/jobs", json=job_request(third, models=LONG_MODELS),
                                    headers=auth(third["capability"]))
        elapsed = time.monotonic() - started
        self.assertEqual(response.status_code, 429, response.text)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(response.json()["error"]["code"], "QUEUE_FULL")
        self.assertGreaterEqual(int(response.headers["retry-after"]), 1)
        self.assertEqual(response.json()["error"]["details"]["retry_after_seconds"], int(response.headers["retry-after"]))
        # the queued job knows its place; cancelling it removes it and leaves the running job alone
        self.assertEqual(self.status(queued)["queue_position"], 1)
        self.assertEqual(self.status(queued)["state"], "queued")
        cancelled = self.cancel(queued)
        self.assertEqual((cancelled.status_code, cancelled.json()["state"]), (200, "cancelled"))
        self.assertIn(self.status(running)["state"], ("queued", "running"))
        # a slot is free again: the third dataset is accepted now
        accepted = self.submit(third, models=LONG_MODELS)
        # cancel everything over HTTP, twice (idempotent)
        for job in (running, accepted):
            wait_for(lambda job=job: self.status(job)["state"] == "running", 90, 0.1, "the job to start")
            first_cancel = self.cancel(job)
            self.assertEqual((first_cancel.status_code, first_cancel.json()["state"]), (200, "cancelled"))
            self.assertEqual(self.cancel(job).json()["state"], "cancelled")
            self.assertEqual(self.client.get(f"/api/v1/jobs/{job['job_id']}/result",
                                             headers=auth(job["capability"])).status_code, 409)
        self.assertNoChildren()
        self.assertEqual(self.coordinator.stats()["heavy_running"], 0)

    def test_a_second_live_job_on_the_same_dataset_is_a_429_and_a_finished_one_is_not(self):
        dataset = self.make_dataset()
        live = self.submit(dataset, models=LONG_MODELS)
        response = self.client.post("/api/v1/jobs", json=job_request(dataset, models=FAST_MODELS),
                                    headers=auth(dataset["capability"]))
        self.assertEqual(response.status_code, 429)
        error = response.json()["error"]
        self.assertEqual(error["code"], "DATASET_BUSY")
        self.assertIn("retry-after", response.headers)
        self.assertEqual(self.cancel(live).json()["state"], "cancelled")
        again = self.submit(dataset, models=["logistic"])
        self.assertEqual(again["state"], "queued")
        self.assertEqual(self.wait_state(again, ("succeeded", "partial", "failed"))["state"], "succeeded")

    def test_cancelling_a_job_that_is_fitting_kills_its_workers(self):
        dataset = self.make_dataset()
        job = self.submit(dataset, models=LONG_MODELS)
        self.wait_fitting(job)
        # another visitor, and a caller with the dataset capability or no capability, cannot cancel it
        stranger = self.make_dataset()
        for token in (stranger["capability"], dataset["capability"], "", "x" * 43):
            denied = self.client.delete(f"/api/v1/jobs/{job['job_id']}", headers=auth(token))
            self.assertEqual((denied.status_code, denied.json()["error"]["code"]), (404, "NOT_FOUND"))
        self.assertEqual(self.status(job)["state"], "running")
        self.assertEqual(self.client.delete(f"/api/v1/datasets/{dataset['dataset_id']}",
                                            headers=auth(stranger["capability"])).status_code, 404)
        self.assertEqual(self.status(job)["state"], "running")
        started = time.monotonic()
        response = self.cancel(job)
        self.assertEqual(response.json()["state"], "cancelled")
        self.assertLess(time.monotonic() - started, 20)
        self.assertNoChildren()
        workers = list(self.coordinator.store.root.rglob("rare-event-worker-*"))
        self.assertEqual(workers, [])


# ============================================================================================ the real server

class V1Process(unittest.TestCase):
    """``python backend/api.py`` as a real process on a free port."""

    @classmethod
    def setUpClass(cls):
        cls.base = support.private_directory()
        cls.marker = cls.base.name
        cls.server = support.ServerProcess(cls.base, {"FYD_UPLOAD_LIMIT_BYTES": "1000000"})
        cls.secrets: list = []
        cls.stopped = False
        try:
            cls.server.wait_ready()
        except BaseException:
            cls.server.stop()
            support.remove_directory(cls.base)
            raise
        cls.http = httpx.Client(base_url=cls.server.url, timeout=60)

    @classmethod
    def tearDownClass(cls):
        problems = []
        cls.http.close()
        code = cls.server.stop()
        if not cls.server.stopped_cleanly(code):
            problems.append(f"the server did not shut down cleanly (exit code {code})")
        if cls.server.root.exists():
            problems.append("the job storage root still exists after shutdown")
        leftover = processes_with(cls.marker)
        for pid in leftover:
            support.kill_pid(pid)
        if leftover:
            problems.append(f"{len(leftover)} process(es) still running after shutdown")
        output = cls.server.output()
        for secret in cls.secrets:
            if secret in output:
                problems.append("a capability appears in the server output")
        if "Traceback" in output:
            problems.append("a traceback appears in the server output")
        support.remove_directory(cls.base)
        if cls.base.exists():
            problems.append("the private test directory could not be removed")
        if problems:
            raise AssertionError("; ".join(problems))

    def dataset(self, rows: int = 300) -> dict:
        response = self.http.post("/api/v1/datasets", content=synthetic_csv(rows=rows, positives=max(40, rows // 8)),
                                  headers={"Content-Type": "text/csv"})
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.secrets.append(body["capability"])
        return body

    def test_the_server_answers_health_head_and_version(self):
        self.assertEqual(self.http.get("/healthz").text, "ok")
        head = self.http.head("/healthz")
        self.assertEqual((head.status_code, head.content), (200, b""))
        self.assertEqual(head.headers["content-length"], "2")
        self.assertEqual(self.http.head("/health").status_code, 200)
        response = self.http.get("/api/v1/version")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("uvicorn", response.headers.get("server", "").lower())   # server_header is off
        self.assertEqual(response.headers["cache-control"], "no-store")
        text = response.text
        for forbidden in (str(support.ROOT), str(self.base), "C:\\"):
            self.assertNotIn(forbidden, text)

    def test_a_chunked_upload_with_no_content_length_is_stopped_with_a_413(self):
        sent = {"bytes": 0}

        def chunks():
            for _ in range(80):   # 4 MB, limit 1 MB
                sent["bytes"] += 50_000
                yield b"y" * 50_000
        started = time.monotonic()
        response = self.http.post("/api/v1/datasets", content=chunks(), headers={"Content-Type": "text/csv"})
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(response.json()["error"]["code"], "REQUEST_TOO_LARGE")
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(self.http.get("/healthz").status_code, 200)
        # with a declared length the answer comes before the body is read
        response = self.http.post("/api/v1/datasets", content=b"z" * 1_200_000, headers={"Content-Type": "text/csv"})
        self.assertEqual(response.status_code, 413)

    def test_health_stays_fast_while_a_real_fit_is_running_and_cancel_leaves_nothing(self):
        dataset = self.dataset()
        headers = auth(dataset["capability"])
        response = self.http.post("/api/v1/jobs", json=job_request(dataset, models=LONG_MODELS), headers=headers)
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.secrets.append(job["capability"])
        job_headers = auth(job["capability"])
        url = f"/api/v1/jobs/{job['job_id']}"

        def fitting():
            return (self.http.get(url, headers=job_headers).json().get("progress") or {}).get("current_model")
        wait_for(fitting, 120, 0.2, "a model to be fitted")
        rounds = []
        passed = False
        for _ in range(4):   # generous retry: a busy machine can delay one round, a hung loop delays all of them
            samples = []
            for _ in range(20):
                started = time.perf_counter()
                reply = self.http.get("/healthz")
                samples.append(time.perf_counter() - started)
                self.assertEqual((reply.status_code, reply.text), (200, "ok"))
                time.sleep(0.05)
            still_fitting = self.http.get(url, headers=job_headers).json()["state"] == "running"
            samples.sort()
            rounds.append({"median_ms": round(samples[len(samples) // 2] * 1000, 1),
                           "p90_ms": round(samples[int(len(samples) * 0.9)] * 1000, 1),
                           "max_ms": round(samples[-1] * 1000, 1), "job_still_running": still_fitting})
            if still_fitting and samples[int(len(samples) * 0.9)] < 0.25:
                passed = True
                break
        print(f"\n[health during a real fit] {rounds}")
        self.assertTrue(passed, rounds)
        # status polling is also served while the fit runs
        started = time.perf_counter()
        self.assertEqual(self.http.get(url, headers=job_headers).status_code, 200)
        self.assertLess(time.perf_counter() - started, 1.0)
        cancelled = self.http.delete(url, headers=job_headers)
        self.assertEqual(cancelled.json()["state"], "cancelled")
        wait_for(lambda: not processes_with(self.marker), 20, 1.0, "every child process to be gone")

    def test_stopping_the_server_with_a_running_job_kills_the_workers_and_deletes_the_storage(self):
        base = support.private_directory()
        server = support.ServerProcess(base)
        secrets = []
        try:
            server.wait_ready()
            with httpx.Client(base_url=server.url, timeout=60) as http:
                dataset = http.post("/api/v1/datasets", content=synthetic_csv(), headers={"Content-Type": "text/csv"}).json()
                job = http.post("/api/v1/jobs", json=job_request(dataset, models=LONG_MODELS),
                                headers=auth(dataset["capability"])).json()
                secrets += [dataset["capability"], job["capability"]]
                url = f"/api/v1/jobs/{job['job_id']}"
                wait_for(lambda: (http.get(url, headers=auth(job["capability"])).json().get("progress") or {})
                         .get("current_model"), 120, 0.2, "a model to be fitted")
                self.assertTrue(processes_with(base.name))
            code = server.stop()
            self.assertTrue(server.stopped_cleanly(code), (code, server.output()[-600:]))
            self.assertFalse(server.root.exists())
            self.assertEqual(processes_with(base.name), [])
            output = server.output()
            for secret in secrets:
                self.assertNotIn(secret, output)
            self.assertNotIn("Traceback", output)
        finally:
            server.stop()
            for pid in processes_with(base.name):
                support.kill_pid(pid)
            support.remove_directory(base)


if __name__ == "__main__":
    unittest.main()
