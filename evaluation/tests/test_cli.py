from __future__ import annotations

import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rare_event_eval.cli import main


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.csv"
        self.input.write_text("outcome,one,two\nclear,0.1,0.2\nevent,0.9,0.8\n", encoding="utf-8")

    def args(self, *extra):
        return ["evaluate", str(self.input), "--label-column", "outcome", "--score-column", "one",
                "--positive-label", "event", "--negative-label", "clear", "--score-kind", "probability", *extra]

    def invoke(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(self.args(*extra))
        return code, out.getvalue(), err.getvalue()

    def test_full_command_writes_json_html_and_input_checksum(self):
        output, html = self.root / "result.json", self.root / "report.html"
        original = self.input.read_bytes()
        code, stdout, _ = self.invoke("--score-column", "two", "--review-budget", "1", "--output", str(output), "--html", str(html))
        self.assertEqual(code, 0)
        self.assertEqual(stdout, "")
        report = json.loads(output.read_text())
        self.assertEqual(len(report["models"]), 2)
        self.assertEqual(report["data"]["input_sha256"], sha256(original).hexdigest())
        self.assertEqual(self.input.read_bytes(), original)
        self.assertIn("<!doctype html>", html.read_text())
        self.assertNotIn(str(self.input), output.read_text())

    def test_stdout_is_valid_json(self):
        code, out, _ = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["data"]["rows"], 2)

    def test_existing_output_and_input_are_not_overwritten(self):
        occupied = self.root / "existing.html"
        occupied.write_text("preserve me")
        new_output = self.root / "new.json"
        code, _, _ = self.invoke("--output", str(new_output), "--html", str(occupied))
        self.assertEqual(code, 2)
        self.assertFalse(new_output.exists())
        self.assertEqual(occupied.read_text(), "preserve me")
        original = self.input.read_bytes()
        self.assertEqual(self.invoke("--output", str(self.input))[0], 2)
        self.assertEqual(self.input.read_bytes(), original)

    def test_same_output_paths_are_rejected(self):
        target = self.root / "both.json"
        self.assertEqual(self.invoke("--output", str(target), "--html", str(target))[0], 2)
        self.assertFalse(target.exists())

    def test_missing_class_option_is_an_argument_error(self):
        args = self.args()
        index = args.index("--positive-label")
        del args[index:index + 2]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            main(args)
        self.assertEqual(result.exception.code, 2)

    def test_bad_csv_rejected_without_exposing_cell_contents(self):
        cases = ["outcome,one\nevent,secret-value\nclear,0.1\n",
                 "outcome,one,one\nevent,.9,.9\n",
                 "outcome,one\nevent,.9,extra\n",
                 "outcome,one\nevent,.9\n\nclear,.1\n",
                 "outcome,one\nthird,.9\nclear,.1\n"]
        for text in cases:
            with self.subTest(text=text):
                self.input.write_text(text, encoding="utf-8")
                code, out, err = self.invoke()
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertNotIn("secret-value", err)

    def test_utf8_bom_supported_and_other_encoding_rejected(self):
        self.input.write_text("outcome,one\nclear,.1\nevent,.9\n", encoding="utf-8-sig")
        self.assertEqual(self.invoke()[0], 0)
        self.input.write_bytes(b"outcome,one\n\xff,.9\n")
        self.assertEqual(self.invoke()[0], 2)

    def test_partial_cost_assumptions_rejected(self):
        self.assertEqual(self.invoke("--false-positive-cost", "2")[0], 2)
        self.assertEqual(self.invoke("--review-cost", "1")[0], 2)

    def test_repeat_columns_and_label_as_score_are_rejected(self):
        self.assertEqual(self.invoke("--score-column", "one")[0], 2)
        self.assertEqual(self.invoke("--score-column", "outcome")[0], 2)

    def test_installed_module_entry_point(self):
        process = subprocess.run([sys.executable, "-B", "-m", "rare_event_eval", "--version"],
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "0.3.0")


if __name__ == "__main__":
    unittest.main()
