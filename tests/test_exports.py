"""Behaviour tests for the job exports, the reproducibility bundle and its replay command.

A real app and coordinator run real jobs on a small synthetic table. The tests check the export allowlist, the
provenance label, the zip (member list, order, timestamps, determinism, size bound), that no row of the upload is in any
export, that every file in the bundle matches the corresponding export, that hostile labels, column names and file
names are escaped, and that the replay command in the bundle is a real command: the command is read from
``replay.txt``, run with the package command line on the original file, and its aggregate results are compared with
the hosted job's. The only differences allowed are the ones listed in ``VOLATILE`` and documented in ``docs/API.md``.

Run from the repository root (the replay tests take about a minute)::

    python -B -m unittest discover -s tests -p "test_exports.py" -v
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import api_support as support
from api_support import ApiCase, auth, job_request, synthetic_csv

from mixed_support import mixed_table, table_to_csv  # noqa: E402
from rare_event_eval import cli  # noqa: E402

import services  # noqa: E402
from services import exports  # noqa: E402

RELEASE = "rel-test-1"
HOSTILE_POSITIVE = "fraud <b>&</b>"
BIDI = chr(0x202E)   # right-to-left override: must show as a marker in the HTML and survive in the JSON
HOSTILE_FILENAME = 'x"><svg onload=alert(1)>.csv'   # an HTTP header holds Latin-1 only, so no bidi control here

# Fields that legitimately differ between the hosted job and the command line replay (see docs/API.md): creation time,
# timings and memory, the parsing record (the app reads uploads with its own table reader), the table limits echoed
# back (the command line cannot set the byte and row limits), and the two labels the app adds.
VOLATILE = [
    r".*/created_utc", r".*/(fit_seconds|fit_and_evaluation_seconds|peak_process_rss_bytes|worker_wall_seconds)",
    r"/input_parsing(/.*)?", r".*/limits/(max_rows|max_bytes)", r"/(provenance|app)",
]


def diff_paths(left, right, path=""):
    """Paths at which two JSON values differ (a path present on one side only counts)."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                found.append(f"{path}/{key}")
            else:
                found += diff_paths(left[key], right[key], f"{path}/{key}")
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [item for index, (a, b) in enumerate(zip(left, right)) for item in diff_paths(a, b, f"{path}[{index}]")]
    return [] if left == right else [path or "/"]


def unexplained(paths):
    return [item for item in paths if not any(re.fullmatch(pattern, item) for pattern in VOLATILE)]


def read_zip(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def hostile_table() -> dict:
    columns = mixed_table(n=240, positives=36, with_row_id=True, with_group=True, group_size=4, seed=11)
    labels = []
    for index, value in enumerate(columns["label"]):
        labels.append(HOSTILE_POSITIVE if value == "fraud" else ("maybe" if index % 3 == 0 else "ok"))
    for index in (7, 100, 200):
        labels[index] = ""   # missing target cells: excluded and listed when the policy is drop_reported
    columns["label"] = labels
    columns["row_id"] = [f"R{index:04d}" for index in range(len(labels))]   # text ids: the app's reader drops a column of bare row numbers
    names = {"row_id": "row id", "age": "age <years>" + BIDI, "income": "income & tax", "color": "colo'r", "city": "$(city)",
             "entity": "grp&1", "label": "outcome <b>?"}
    return {names.get(key, key): value for key, value in columns.items()}


HOSTILE_PARAMS = {"target": "outcome <b>?", "positive_label": HOSTILE_POSITIVE, "one_vs_rest": True,
                  "models": ["logistic", "rf_weighted"], "split_mode": "group", "group_column": "grp&1",
                  "id_columns": ["row id"], "exclude_columns": ["$(city)"], "missing_policy": "drop_reported",
                  "seed": 5, "review_fractions": [0.1, 0.25],
                  "costs": {"false_positive_cost": 2, "false_negative_cost": 15}}
PLAIN_PARAMS = {"models": ["logistic", "rf_weighted"], "calibration": "sigmoid", "review_fractions": [0.05, 0.1, 0.3],
                "costs": {"false_positive_cost": 1.0, "false_negative_cost": 8.0, "review_cost": 0.25},
                "exclude_columns": ["city"], "seed": 7}


def api_version() -> str:
    import api
    return api.APP_VERSION


def replay_lines(members: dict) -> tuple:
    """``(posix command line, powershell line or None, json argument list)`` read from the text of replay.txt."""
    text = members["replay.txt"].decode("utf-8")
    lines = text.splitlines()
    posix = next(line for line in lines if line.startswith("       python -m rare_event_eval")).strip()
    arguments = json.loads(next(line for line in lines if line.startswith("       [")).strip())
    shell = next((line.strip() for line in lines[lines.index("   Windows PowerShell:"):]
                  if line.startswith("       python -m")), None)
    return posix, shell, arguments


class ExportCase(ApiCase):
    """Runs each scenario's job once (lazily) and shares it between the tests of the class."""

    @classmethod
    def setUpClass(cls):
        cls._env = mock.patch.dict(os.environ, {"FYD_RELEASE_ID": RELEASE})
        cls._env.start()
        super().setUpClass()
        cls.runs: dict = {}

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            cls._env.stop()

    def run_job(self, key: str, raw: bytes, body: dict, filename: str = "table.csv") -> dict:
        if key in self.runs:
            return self.runs[key]
        response = self.client.post("/api/v1/datasets", content=raw,
                                    headers={"Content-Type": "text/csv", "X-Filename": filename})
        self.assertEqual(response.status_code, 201, response.text)
        dataset = response.json()
        self.remember(dataset["capability"])
        response = self.client.post("/api/v1/jobs", json={**body, "dataset_id": dataset["dataset_id"]},
                                    headers=auth(dataset["capability"]))
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.remember(job["capability"])
        type(self).jobs.append(job)
        status = self.wait_state(job, ("succeeded", "partial", "failed", "cancelled", "timed_out"), timeout=240)
        run = {"dataset": dataset, "job": job, "raw": raw, "status": status, "body": body}
        self.runs[key] = run
        return run

    def plain(self) -> dict:
        run = self.run_job("plain", synthetic_csv(), job_request({"dataset_id": "x"}, **PLAIN_PARAMS))
        self.assertEqual(run["status"]["state"], "succeeded", run["status"])
        return run

    def hostile(self) -> dict:
        body = {key: value for key, value in HOSTILE_PARAMS.items()}
        run = self.run_job("hostile", table_to_csv(hostile_table()), body, HOSTILE_FILENAME)
        self.assertEqual(run["status"]["state"], "succeeded", run["status"])
        return run

    def export(self, run: dict, kind: str, token: str | None = None):
        return self.client.get(f"/api/v1/jobs/{run['job']['job_id']}/exports/{kind}",
                               headers=auth(token or run["job"]["capability"]))

    def result(self, run: dict) -> dict:
        response = self.client.get(f"/api/v1/jobs/{run['job']['job_id']}/result", headers=auth(run["job"]["capability"]))
        self.assertEqual(response.status_code, 200)
        return response.json()


class ReplayAndSettingsHelpers(unittest.TestCase):
    """Pure functions: no job runs here."""

    def test_the_defaults_in_the_effective_settings_are_the_services_own(self):
        defaults = services.effective_settings({"target": "t", "positive_label": "p", "negative_label": "n"},
                                               {"max_bytes": 1, "max_rows": 2, "max_expanded_features": 3}, 60)
        self.assertEqual(defaults["models"], list(services.app_default_models()))
        self.assertEqual((defaults["split_mode"], defaults["seed"], defaults["calibration"], defaults["missing_policy"]),
                         ("stratified", 42, "none", "error"))
        self.assertEqual(defaults["review_fractions"], [0.1, 0.2])

    def test_the_label_set_matches_the_package(self):
        from rare_event_eval.reports import PROVENANCE_LABELS
        self.assertEqual(exports.PROVENANCE_LABELS, PROVENANCE_LABELS)

    def test_notes_cover_conversion_warnings_size_failures_and_unsupported_values(self):
        settings = services.effective_settings(PLAIN_PARAMS | {"target": "t", "positive_label": "p", "negative_label": "n"},
                                               {"max_bytes": 8_000_000, "max_rows": 10000, "max_expanded_features": 100}, 60)
        report = {"input_parsing": {"details": {"format_notes": ["Converted from Excel"], "read_warnings": ["Skipped 2 malformed rows"],
                                                "input_bytes": 6_000_000}},
                  "failures": [{"model": "erf", "reason_code": "MODEL_TIMEOUT"}]}
        notes = exports.replay_notes(report, settings, ["split mode 'x' is not a command line choice"])
        cannot, caution = " ".join(notes["cannot"]), " ".join(notes["caution"])
        self.assertIn("split mode 'x'", cannot)
        self.assertIn("refuses files above 5000000 bytes", cannot)
        self.assertIn("Converted from Excel", caution)
        self.assertIn("Skipped 2 malformed rows", caution)
        self.assertIn("timeout depends on the speed of the machine", caution)
        clean = exports.replay_notes({"input_parsing": {"details": {"input_bytes": 100}}, "failures": []}, settings, [])
        self.assertEqual(clean["caution"], [])
        # a setting the command line cannot express is reported, never silently dropped
        odd = dict(settings, split_mode="time_blocks", calibration="isotonic")
        _, unsupported = exports.replay_argv(odd)
        self.assertEqual(len(unsupported), 2)

    def test_optional_settings_become_the_matching_flags(self):
        base = {"target": "t", "positive_label": 1, "negative_label": 0}
        limits = {"max_bytes": 1, "max_rows": 1, "max_expanded_features": 40}
        args, _ = exports.replay_argv(services.effective_settings(
            base | {"split_mode": "chronological", "time_column": "when", "id_columns": ["id"],
                    "missing_policy": "drop_reported", "models": ["erf"], "review_fractions": [0.5]}, limits, 12.5))
        self.assertIn("--positive-label=1", args)
        self.assertIn("--negative-label=0", args)
        for flag in ("--split=chronological", "--time-column=when", "--id-column=id", "--missing-target=drop_reported",
                     "--model=erf", "--review-fraction=0.5", "--timeout=12.5", "--max-expanded-features=40"):
            self.assertIn(flag, args)
        self.assertFalse(any(item.startswith(("--false-positive-cost", "--group-column")) for item in args))
        ovr, _ = exports.replay_argv(services.effective_settings(
            {"target": "t", "positive_label": "p", "one_vs_rest": True, "group_column": "g",
             "split_mode": "group"}, limits, 60))
        self.assertIn("--one-vs-rest", ovr)
        self.assertFalse(any(item.startswith("--negative-label") for item in ovr))
        self.assertIn("--group-column=g", ovr)
        parsed = cli._parser().parse_args(ovr[3:])
        self.assertTrue(parsed.one_vs_rest and parsed.group_column == "g")

    def test_quoting_survives_hostile_values(self):
        values = ["fraud <b>&</b>", "it's", "$(whoami)", "`tick`", "a b", "x;y", "-leading", "--model=erf", "\u00e9",
                  "100%", "@splat", "(paren)", "{brace}", "#hash", "a,b", "a|b", "*", "~", "!bang", "back\\slash", "say \"hi\""]
        for value in values:
            args = ["python", "-m", "rare_event_eval", "train", "in.csv", f"--target={value}", f"--positive-label={value}"]
            self.assertEqual(shlex.split(exports.posix_quote(args)), args, value)
            shell = exports.powershell_quote(args)
            self.assertEqual(shell is None, '"' in value, value)
        controls, _ = exports.replay_argv(services.effective_settings(
            {"target": "line\nbreak", "positive_label": "p", "negative_label": "n"},
            {"max_expanded_features": 10}, 60))
        report = {"input_sha256": "0" * 64, "input_parsing": {"details": {}}}
        text = exports.replay_text(report, services.effective_settings(
            {"target": "line\nbreak", "positive_label": "p", "negative_label": "n"}, {"max_expanded_features": 10}, 60),
            {"source_release_id": "x", "package": {"version": "1"}})
        self.assertIn("control character", text)
        self.assertNotIn("POSIX shell (bash, zsh):", text)
        self.assertIn(json.dumps(controls), text)   # the exact argument list is still given

    @unittest.skipUnless(os.name == "nt", "Windows PowerShell quoting is checked on Windows only")
    def test_the_powershell_command_passes_hostile_values_unchanged(self):
        values = ["fraud <b>&</b>", "it's", "$(whoami)", "`tick`", "a b", "x;y", "-leading", "\u00e9", "100%", "@splat",
                  "(paren)", "{brace}", "#hash", "a,b", "a|b", "*", "~", "!bang", "back\\slash", "plain"]
        with tempfile.TemporaryDirectory(prefix="fyd-ps-test-") as folder:
            script = Path(folder) / "argdump.py"
            script.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
            args = ["python", "-m", "rare_event_eval", "train", "in.csv"] + [f"--target={value}" for value in values]
            rest = exports.powershell_quote(["x"] + args[4:])[2:]
            command = f"& '{sys.executable}' '{script}' {rest}"
            done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                  capture_output=True, text=True, timeout=120, encoding="utf-8")
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout.strip().splitlines()[-1]), args[4:])


class Exports(ExportCase):
    """Allowlist, labels, bundle, replay text and the replay runs (each scenario's job runs once)."""

    # ---- allowlist, provenance, config

    def test_the_four_kinds_have_their_types_names_and_headers(self):
        run = self.plain()
        expected = {"report_json": ("application/json", "fyd-report.json"),
                    "report_html": ("text/html; charset=utf-8", "fyd-report.html"),
                    "config_json": ("application/json", "fyd-config.json"),
                    "repro_bundle": ("application/zip", "fyd-repro-bundle.zip")}
        for kind, (content_type, name) in expected.items():
            response = self.export(run, kind)
            self.assertEqual(response.status_code, 200, kind)
            self.assertEqual(response.headers["content-type"], content_type, kind)
            self.assertEqual(response.headers["content-disposition"], f'attachment; filename="{name}"', kind)
            self.assertEqual(response.headers["cache-control"], "no-store", kind)
            self.assertEqual(response.headers["x-content-type-options"], "nosniff", kind)
            self.assertEqual(("content-security-policy" in response.headers), kind == "report_html", kind)

    def test_everything_else_is_an_unknown_export_and_no_row_level_export_exists(self):
        run = self.plain()
        for kind in ("report", "repro_bundle.zip", "REPRO_BUNDLE", "bundle", "rows", "row_mappings", "scores",
                     "split_indices", "predictions", "data", "..%2freport_json", "report_json%00", "config"):
            response = self.export(run, kind)
            self.assertEqual(response.status_code, 404, kind)
            self.assertIn(response.json()["error"]["code"], ("EXPORT_UNKNOWN", "NOT_FOUND"), kind)
        self.assertEqual(exports.BUNDLE_FILES, ("config.json", "report.json", "report.html", "versions.json",
                                                "README.txt", "replay.txt"))

    def test_a_missing_wrong_or_foreign_capability_is_the_same_404(self):
        run = self.plain()
        other = run["dataset"]["capability"]   # a real capability, but not this job's
        for token in ("", "wrong", other):
            response = self.client.get(f"/api/v1/jobs/{run['job']['job_id']}/exports/repro_bundle",
                                       headers=auth(token) if token else {})
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["error"]["code"], "NOT_FOUND")

    def test_a_job_without_a_report_has_no_export(self):
        run = self.run_job("failed", synthetic_csv(), job_request({"dataset_id": "x"}, target="no_such_column"))
        self.assertEqual(run["status"]["state"], "failed")
        for kind in ("report_json", "report_html", "config_json", "repro_bundle"):
            response = self.export(run, kind)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (409, "RESULT_NOT_AVAILABLE"), kind)

    def test_result_and_report_json_are_the_same_labelled_report(self):
        run = self.plain()
        result = self.result(run)
        exported = self.export(run, "report_json")
        self.assertEqual(json.loads(exported.content), result)
        self.assertEqual(result["provenance"], "fresh app run")
        self.assertEqual(result["app"]["release_id"], RELEASE)
        self.assertEqual(result["app"]["app_version"], api_version())
        self.assertEqual(result["schema_version"], "training-1.1")
        for field in ("label_mapping", "split", "model_records", "outcomes", "input_sha256"):   # nothing was taken away
            self.assertIn(field, result)
        self.assertEqual(self.export(run, "report_json").content, exported.content)   # stable bytes
        for raw in (exported.content, self.export(run, "config_json").content):
            self.assertTrue(raw.isascii())
            for char in (b"<", b">", b"&"):
                self.assertNotIn(char, raw)
        self.assertNotRegex(exported.text, r"\b(NaN|Infinity)\b")

    def test_report_html_states_the_provenance_and_the_assumptions_wording(self):
        run = self.plain()
        html = self.export(run, "report_html").text
        self.assertIn("<strong>fresh app run</strong>", html)
        self.assertIn(f"source release {RELEASE}", html)
        self.assertIn("assumptions, not measured savings", html)
        self.assertIn("User-supplied tabular training data", html)   # a fresh hosted run keeps this wording
        self.assertNotIn("seeded script", html)
        self.assertIn("Validation_only".lower(), html.lower())
        self.assertNotRegex(html, r"https?://")
        self.assertNotIn("<script", html.lower())

    def test_config_json_writes_every_default_out(self):
        run = self.plain()
        config = json.loads(self.export(run, "config_json").content)
        self.assertEqual(config["schema"], "fyd-job-config-1.0")
        self.assertEqual(config["provenance"], "fresh app run")
        self.assertEqual(config["request"]["models"], ["logistic", "rf_weighted"])
        effective = config["effective"]
        self.assertEqual((effective["split_mode"], effective["calibration"], effective["missing_policy"], effective["seed"]),
                         ("stratified", "sigmoid", "error", 7))
        self.assertEqual(effective["review_fractions"], [0.05, 0.1, 0.3])
        self.assertEqual(effective["exclude_columns"], ["city"])
        self.assertEqual(effective["costs"], {"false_positive_cost": 1.0, "false_negative_cost": 8.0, "review_cost": 0.25})
        self.assertEqual(effective["table_limits"]["max_expanded_features"], 100)
        self.assertEqual(config["input_sha256"], self.result(run)["input_sha256"])
        self.assertNotIn("capability", json.dumps(config))

    # ---- bundle contents

    def test_the_zip_has_the_six_files_in_a_fixed_order_with_fixed_timestamps(self):
        run = self.plain()
        data = self.export(run, "repro_bundle").content
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.namelist(), list(exports.BUNDLE_FILES))
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0), info.filename)
                self.assertEqual(info.external_attr >> 16, 0o644)
                self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)
                self.assertFalse(info.flag_bits & 0x1)   # no encryption
                self.assertEqual(info.filename, os.path.basename(info.filename))   # no folders, no path tricks
        self.assertLess(len(data), exports.MAX_BUNDLE_BYTES)

    def test_two_exports_of_one_job_are_byte_identical(self):
        run = self.plain()
        self.assertEqual(self.export(run, "repro_bundle").content, self.export(run, "repro_bundle").content)
        report = self.result(run)
        settings = services.effective_settings(PLAIN_PARAMS | {"target": "label", "positive_label": "fraud",
                                                               "negative_label": "ok"},
                                               {"max_expanded_features": 100, "max_bytes": 1_000_000, "max_rows": 10000}, 60.0)
        again = [exports.build_repro_bundle(report, settings, {"a": 1}, "<p>x</p>", {"app_version": "1"}) for _ in range(2)]
        self.assertEqual(again[0], again[1])

    def test_members_match_the_other_exports(self):
        run = self.plain()
        members = read_zip(self.export(run, "repro_bundle").content)
        self.assertEqual(members["report.json"], self.export(run, "report_json").content)
        self.assertEqual(members["report.html"], self.export(run, "report_html").content)
        bundled = json.loads(members["config.json"])
        self.assertEqual(bundled.pop("bundle_schema"), "fyd-repro-bundle-1.0")
        self.assertEqual(bundled, json.loads(self.export(run, "config_json").content))
        for name in ("config.json", "report.json", "versions.json"):
            self.assertTrue(members[name].isascii(), name)
            self.assertTrue(members[name].endswith(b"\n"), name)
            for char in (b"<", b">", b"&"):
                self.assertNotIn(char, members[name], name)

    def test_versions_json_records_app_package_schema_python_libraries_recipes_and_release(self):
        run = self.plain()
        versions = json.loads(read_zip(self.export(run, "repro_bundle").content)["versions.json"])
        report = self.result(run)
        self.assertEqual(versions["schema"], "fyd-versions-1.0")
        self.assertEqual(versions["app_version"], api_version())
        self.assertEqual(versions["source_release_id"], RELEASE)
        self.assertEqual(versions["report_schema"], "training-1.1")
        self.assertEqual(versions["job_config_schema"], "fyd-job-config-1.0")
        tool = next(row["report"]["tool"] for row in report["outcomes"].values())
        self.assertEqual(versions["package"], {"name": "rare-event-eval", "version": tool["version"]})
        self.assertEqual(versions["python"], tool["python"])
        self.assertEqual(versions["libraries"]["numpy"], tool["numpy"])
        self.assertEqual(versions["libraries"]["scikit_learn"], tool["scikit_learn"])
        self.assertTrue(versions["libraries"]["imbalanced_learn"] and versions["libraries"]["pandas"])
        self.assertEqual(set(versions["recipes"]), {"logistic", "rf_weighted"})
        for requested, item in versions["recipes"].items():
            self.assertEqual(item["recipe_id"], report["model_records"][requested]["recipe_id"])
            self.assertRegex(item["recipe_id"], r"^rare_event_eval/[a-z_0-9-]+/\d+$")
        self.assertNotRegex(json.dumps(versions), r"[A-Za-z]:\\|/Users/|" + re.escape(Path.home().name))

    def test_no_row_of_the_upload_is_in_any_file(self):
        run = self.plain()
        members = read_zip(self.export(run, "repro_bundle").content)
        rows = run["raw"].decode("utf-8").splitlines()[1:]
        everything = b"\n".join(members.values()).decode("utf-8")
        for row in rows[:60] + rows[-20:]:
            self.assertNotIn(row, everything)
        values = [cell for row in rows[:40] for cell in row.split(",") if re.fullmatch(r"-?\d+\.\d{3}", cell)]
        leaked = [cell for cell in values if re.search(rf"(?<![\d.]){re.escape(cell)}(?![\d])", everything)]
        self.assertLessEqual(len(leaked), 2, leaked)   # a short decimal can occur by chance in a metric; rows cannot
        self.assertEqual(list(members), list(exports.BUNDLE_FILES))   # no data file, no model file, no index file
        for name in ("split_indices", "row_mappings", "predictions", "scores", "indices.csv"):
            self.assertFalse(any(name in member for member in members))

    def test_readme_states_what_is_and_is_not_inside(self):
        run = self.plain()
        readme = read_zip(self.export(run, "repro_bundle").content)["README.txt"].decode("utf-8")
        for text in ("Provenance of the result: fresh app run.", "What is not in this bundle", "uploaded table",
                     "any prediction or score per row", "row indexes", "fitted model", "not a historical research result",
                     f"Source release: {RELEASE}", "config.json", "report.json", "report.html", "versions.json", "replay.txt"):
            self.assertIn(text, readme)

    def test_the_bundle_is_bounded(self):
        run = self.plain()
        with mock.patch.object(exports, "MAX_BUNDLE_UNCOMPRESSED_BYTES", 2000):
            response = self.export(run, "repro_bundle")
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (500, "EXPORT_TOO_LARGE"))
        with mock.patch.object(exports, "MAX_BUNDLE_BYTES", 2000):
            response = self.export(run, "repro_bundle")
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (500, "EXPORT_TOO_LARGE"))
        self.assertEqual(self.export(run, "repro_bundle").status_code, 200)   # and fine again afterwards

    # ---- replay text

    def test_the_command_is_the_cli_with_every_setting_and_parses(self):
        run = self.plain()
        members = read_zip(self.export(run, "repro_bundle").content)
        posix, shell, arguments = replay_lines(members)
        self.assertEqual(shlex.split(posix), arguments)   # the three spellings are one command
        self.assertEqual(arguments[:5], ["python", "-m", "rare_event_eval", "train", exports.INPUT_PLACEHOLDER])
        parsed = cli._parser().parse_args(arguments[3:])   # the real parser accepts every flag
        self.assertEqual((parsed.command, parsed.target, parsed.positive_label, parsed.negative_label, parsed.one_vs_rest),
                         ("train", "label", "fraud", "ok", False))
        self.assertEqual((parsed.model, parsed.split, parsed.seed, parsed.calibration, parsed.missing_target),
                         (["logistic", "rf_weighted"], "stratified", 7, "sigmoid", "error"))
        self.assertEqual((parsed.review_fraction, parsed.timeout, parsed.exclude_column, parsed.id_column),
                         ([0.05, 0.1, 0.3], 60.0, ["city"], None))
        self.assertEqual((parsed.false_positive_cost, parsed.false_negative_cost, parsed.review_cost), (1.0, 8.0, 0.25))
        self.assertEqual(parsed.max_expanded_features, 100)
        self.assertEqual((str(parsed.output), str(parsed.html)), (exports.OUTPUT_JSON, exports.OUTPUT_HTML))

    def test_the_text_says_what_it_does_and_does_not_do(self):
        run = self.plain()
        text = read_zip(self.export(run, "repro_bundle").content)["replay.txt"].decode("utf-8")
        report = self.result(run)
        self.assertIn(report["input_sha256"], text)
        self.assertIn("does not contain the data", text)
        self.assertIn("never overwrites an existing file", text)
        self.assertIn("What should match and what may differ", text)
        self.assertIn("What the command cannot express", text)
        self.assertIn("the row limit of the app (10000 rows)", text)
        self.assertIn(f"Source release of the run that made this bundle: {RELEASE}", text)
        self.assertIn(report["outcomes"]["logistic"]["report"]["tool"]["version"], text)
        self.assertNotRegex(text, r"[A-Za-z]:\\|/Users/|" + re.escape(Path.home().name))






    # ---- the replay command, run on the original file, gives the hosted job's aggregate results

    def replay(self, run: dict) -> dict:
        members = read_zip(self.export(run, "repro_bundle").content)
        posix, _, arguments = replay_lines(members)
        argv = shlex.split(posix)
        self.assertEqual(argv, arguments)
        with tempfile.TemporaryDirectory(prefix="fyd-replay-") as folder:
            work = Path(folder)
            (work / "input.csv").write_bytes(run["raw"])
            argv[argv.index("train") + 1] = "input.csv"
            command = [sys.executable, "-B"] + argv[1:]
            done = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=400, encoding="utf-8",
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(done.returncode, 0, done.stderr[-1500:])
            report = json.loads((work / "replay-report.json").read_text(encoding="utf-8"))
            html = (work / "replay-report.html").read_text(encoding="utf-8")
        return {"report": report, "html": html, "members": members}

    def assert_same_aggregates(self, hosted: dict, replayed: dict):
        self.assertEqual(replayed["input_sha256"], hosted["input_sha256"])
        for key in ("label_mapping", "predictors", "predictor_kinds", "excluded_columns", "split", "model_records",
                    "seed", "failures", "fallback_policy"):
            self.assertEqual(replayed[key], hosted[key], key)
        for name, row in hosted["outcomes"].items():
            other = replayed["outcomes"][name]
            self.assertEqual((row["status"], other["status"]), ("success", "success"), name)
            self.assertEqual(other["report"]["models"], row["report"]["models"], name)   # metrics, thresholds, budgets, bins
            self.assertEqual(other["report"]["policies"], row["report"]["policies"], name)
            self.assertEqual(other["execution"]["threshold_selection"], row["execution"]["threshold_selection"], name)
            self.assertEqual(other["model_record"], row["model_record"], name)
        self.assertEqual(replayed["preprocessing"]["record"]["alignment"], hosted["preprocessing"]["record"]["alignment"])
        self.assertEqual(replayed["preprocessing"]["adapter"], hosted["preprocessing"]["adapter"])
        rest = unexplained(diff_paths(hosted, replayed))
        self.assertEqual(rest, [], f"fields that differ and are not documented: {rest[:12]}")

    def test_the_plain_job_is_reproduced_by_its_replay_command(self):
        run = self.plain()
        hosted = self.result(run)
        replayed = self.replay(run)
        self.assert_same_aggregates(hosted, replayed["report"])
        # the documented differences are real and limited to the listed kinds
        differing = diff_paths(hosted, replayed["report"])
        self.assertIn("/provenance", differing)
        self.assertTrue(any(item.startswith("/input_parsing") for item in differing))
        self.assertNotRegex(replayed["html"], r"https?://")   # the replayed page is standalone too
        self.assertIn("not labelled", replayed["html"])   # a command line run carries no app provenance label

    def test_a_hostile_one_versus_rest_group_job_is_reproduced_and_escaped_everywhere(self):
        run = self.hostile()
        hosted = self.result(run)
        self.assertEqual(hosted["label_mapping"]["mode"], "one_vs_rest")
        self.assertEqual(hosted["label_mapping"]["counts"]["missing_target_rows"], 3)
        self.assertEqual(hosted["split"]["achieved"]["mode"], "group")
        replayed = self.replay(run)
        self.assert_same_aggregates(hosted, replayed["report"])
        # every export: the raw hostile text never appears as markup or as an unescaped value
        members = replayed["members"]
        html = members["report.html"].decode("utf-8")
        self.assertEqual(html, self.export(run, "report_html").text)
        for forbidden in ("<b>", "<svg", "<script", "onload=alert(1)>"):
            self.assertNotIn(forbidden, html.lower())
        self.assertIn("age &lt;years&gt;&lt;U+202E&gt;", html)   # the bidi override is a visible marker in the page ...
        self.assertNotIn(BIDI, html)
        self.assertIn(chr(92) + "u202e", members["report.json"].decode("ascii"))   # ... and the real character in the JSON
        self.assertEqual(json.loads(members["report.json"])["predictors"].count("age <years>" + BIDI), 1)
        for escaped in ("fraud &lt;b&gt;&amp;&lt;/b&gt;", "age &lt;years&gt;", "income &amp; tax", "colo&#x27;r", "grp&amp;1"):
            self.assertIn(escaped, html)
        for name in ("config.json", "report.json", "versions.json"):
            for char in (b"<", b">", b"&"):
                self.assertNotIn(char, members[name], name)
        self.assertIn("\\u003cb\\u003e", members["report.json"].decode("ascii"))
        for blob in members.values():
            self.assertNotIn(HOSTILE_FILENAME.encode(), blob)   # the upload's file name is never echoed
            self.assertNotIn(b"onload=alert(1)", blob)
        # the text file keeps the real values, quoted for a shell, so the command works on the real names
        posix, shell, arguments = replay_lines(members)
        self.assertIn(f"--positive-label={HOSTILE_POSITIVE}", arguments)
        self.assertIn("--target=outcome <b>?", arguments)
        self.assertIn("--one-vs-rest", arguments)
        self.assertIn("--group-column=grp&1", arguments)
        self.assertIn("--id-column=row id", arguments)
        self.assertIn("--exclude-column=$(city)", arguments)
        self.assertIn("--missing-target=drop_reported", arguments)
        self.assertIsNotNone(shell)   # no double quote in these values, so the PowerShell command is given too


if __name__ == "__main__":
    unittest.main()
