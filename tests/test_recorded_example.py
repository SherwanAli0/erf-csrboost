"""The recorded example: saved files, labels, honesty of its text, and regeneration to the same aggregate results.

``evaluation/examples/make_recorded_example.py`` runs a seeded synthetic table through the application's job path and
saves the report (JSON and HTML). These tests check the saved files, then run the generator again into a private folder
and require the same table bytes and the same aggregate results; only the creation time, timings and memory may differ.

Run from the repository root (the regeneration test takes about half a minute)::

    python -B -m unittest discover -s tests -p "test_recorded_example.py" -v
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import api_support as support

from rare_event_eval import cli  # noqa: E402
from rare_event_eval.reports import render_html, training_report  # noqa: E402

import services  # noqa: E402

EXAMPLES = support.ROOT / "evaluation" / "examples"
OUTPUT = EXAMPLES / "output"
WEB = support.ROOT / "evaluation" / "src" / "rare_event_eval" / "web"
GENERATOR = EXAMPLES / "make_recorded_example.py"
VOLATILE = [r".*/created_utc", r".*/tool/python", r".*/environment/python", r".*/(fit_seconds|fit_and_evaluation_seconds|peak_process_rss_bytes|worker_wall_seconds)"]


def load_generator():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_recorded_example", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diff_paths(left, right, path=""):
    # BLAS/libm builds can differ by a last floating-point bit across OSes.
    # Keep keys, populations, labels, counts and policies exact; bound floats to 1e-12.
    if isinstance(left,float) and isinstance(right,float):
        return [] if math.isclose(left,right,rel_tol=0,abs_tol=1e-12) else [path or '/']
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


class SavedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.json_bytes = (WEB / "recorded-example.json").read_bytes()
        cls.html_bytes = (WEB / "recorded-example.html").read_bytes()
        cls.report = json.loads(cls.json_bytes)

    def test_the_web_copies_equal_the_files_saved_next_to_the_generator(self):
        self.assertEqual(self.json_bytes, (OUTPUT / "recorded-example.json").read_bytes())
        self.assertEqual(self.html_bytes, (OUTPUT / "recorded-example.html").read_bytes())
        self.assertTrue((EXAMPLES / "recorded_example_data.csv").is_file())

    def test_the_json_is_a_labelled_training_report_and_is_not_hand_edited(self):
        report = self.report
        self.assertEqual((report["schema_version"], report["provenance"]), ("training-1.1", "recorded app run"))
        self.assertEqual(set(report["app"]), {"app_version", "release_id", "note"})
        self.assertEqual(self.json_bytes, services.dump_json(report))   # exactly the canonical writer's output
        self.assertTrue(self.json_bytes.isascii())
        self.assertNotRegex(self.json_bytes.decode("ascii"), r"\b(NaN|Infinity)\b")
        for char in (b"<", b">", b"&"):
            self.assertNotIn(char, self.json_bytes)
        # the html is a pure function of the json: nothing in it was written by hand
        self.assertEqual(services.render_report_html(report).encode("utf-8"), self.html_bytes)
        self.assertEqual(render_html(training_report(report)).encode("utf-8"), self.html_bytes)

    def test_every_model_ran_and_the_example_has_a_competitive_simple_model(self):
        report = self.report
        self.assertEqual(sorted(report["outcomes"]), sorted(["logistic", "rf_weighted", "balanced_rf",
                                                             "hist_gradient_boosting", "erf"]))
        ap = {}
        for name, row in report["outcomes"].items():
            self.assertEqual(row["status"], "success", name)
            ap[name] = row["report"]["models"][name]["metrics"]["average_precision"]
        self.assertEqual({k: round(v, 6) for k, v in ap.items()}, report["example"]["average_precision_on_test"])
        best_other = max(value for name, value in ap.items() if name != "logistic")
        self.assertGreaterEqual(ap["logistic"], best_other - 0.05)   # the simpler model is competitive or better
        self.assertIn("logistic regression", report["example"]["description"])
        self.assertIn(f"{ap['logistic']:.3f}", report["example"]["description"])
        self.assertEqual(report["failures"], [])

    def test_the_example_is_synthetic_documented_and_free_of_paths_and_rows(self):
        text = self.json_bytes.decode("ascii") + self.html_bytes.decode("utf-8")
        self.assertNotRegex(text, r"[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|/tmp/|AppData|" + re.escape(Path.home().name))
        self.assertNotRegex(text, r"C0{2,}\d{2}")   # no customer identifier of the table
        example = self.report["example"]
        self.assertIn("no real person, no external data and nothing confidential", example["data_origin"])
        self.assertIn("additive by construction", example["note"])
        self.assertIn("not measured savings", example["note"])
        self.assertEqual(example["generator"]["data_file"], "evaluation/examples/recorded_example_data.csv")
        self.assertEqual(example["generator"]["regenerate_command"], "python evaluation/examples/make_recorded_example.py")
        self.assertNotRegex(self.html_bytes.decode("utf-8"), r"https?://")

    def test_the_replay_arguments_of_the_example_are_a_valid_command_for_the_data_file(self):
        arguments = self.report["example"]["generator"]["replay_command_arguments"]
        self.assertEqual(arguments[:4], ["python", "-m", "rare_event_eval", "train"])
        self.assertEqual(arguments[4], "evaluation/examples/recorded_example_data.csv")
        parsed = cli._parser().parse_args(arguments[3:])
        self.assertEqual((parsed.target, parsed.positive_label, parsed.negative_label, parsed.id_column, parsed.seed),
                         ("outcome", "default", "repaid", ["customer_id"], 7))
        self.assertEqual(parsed.model, ["logistic", "rf_weighted", "balanced_rf", "hist_gradient_boosting", "erf"])
        self.assertEqual(parsed.review_fraction, [0.05, 0.1, 0.2])

    def test_the_page_names_the_real_source_of_the_table(self):
        page = self.html_bytes.decode("utf-8")
        self.assertIn("Synthetic table generated by a seeded script in the repository.", page)
        self.assertNotIn("User-supplied", page)
        self.assertNotIn("User-declared description", page)

    def test_the_html_names_the_recorded_run_and_states_the_example(self):
        page = self.html_bytes.decode("utf-8")
        self.assertIn("<strong>recorded app run</strong>", page)
        self.assertIn("About this example", page)
        self.assertIn("It is not a historical ERF research result", page)


class Regeneration(unittest.TestCase):
    def test_numeric_comparison_tolerates_roundoff_but_detects_material_changes(self):
        self.assertEqual(diff_paths({'ap':.5},{'ap':.5+1e-13}),[])
        self.assertEqual(diff_paths({'ap':.5},{'ap':.5+1e-8}),['/ap'])
        self.assertEqual(diff_paths({'count':10},{'count':11}),['/count'])

    def test_the_generator_reproduces_the_table_and_the_aggregate_results(self):
        generator = load_generator()
        saved_csv = (EXAMPLES / "recorded_example_data.csv").read_bytes()
        self.assertEqual(generator.synthetic_csv(), saved_csv)   # the table itself is a pure function of the seed
        with tempfile.TemporaryDirectory(prefix="fyd-regen-") as folder:
            done = subprocess.run([sys.executable, "-B", str(GENERATOR), "--into", folder], cwd=str(support.ROOT),
                                  capture_output=True, text=True, timeout=600, encoding="utf-8",
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(done.returncode, 0, done.stderr[-1500:])
            regenerated = json.loads((Path(folder) / "recorded-example.json").read_text(encoding="utf-8"))
            self.assertEqual((Path(folder) / "recorded_example_data.csv").read_bytes(), saved_csv)
            html = (Path(folder) / "recorded-example.html").read_text(encoding="utf-8")
        saved = json.loads((WEB / "recorded-example.json").read_bytes())
        differing = diff_paths(saved, regenerated)
        rest = [item for item in differing if not any(re.fullmatch(pattern, item) for pattern in VOLATILE)]
        self.assertEqual(rest, [], f"fields that changed and are not times, timings or memory: {rest[:10]}")
        for name, row in saved["outcomes"].items():   # the headline numbers, explicitly
            again = regenerated["outcomes"][name]["report"]["models"][name]
            self.assertEqual(diff_paths(again["metrics"], row["report"]["models"][name]["metrics"]), [], name)
            self.assertEqual(diff_paths(again["threshold_result"], row["report"]["models"][name]["threshold_result"]), [], name)
        self.assertEqual(regenerated["example"], saved["example"])
        self.assertEqual(html, services.render_report_html(regenerated))
        leftovers = [item for item in Path(tempfile.gettempdir()).glob("fyd-recorded-example-*")]
        self.assertEqual(leftovers, [])   # the generator removed its private storage folder


if __name__ == "__main__":
    unittest.main()
