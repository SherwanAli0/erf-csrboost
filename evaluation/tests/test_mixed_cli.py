import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mixed_support import mixed_table, numeric_table, patched_runner, table_to_csv
from rare_event_eval.cli import main

try:
    import pandas
except ImportError:  # pragma: no cover - exercised in the venv without pandas
    pandas = None


class TrainCommandRouting(unittest.TestCase):
    def invoke(self, columns, *flags, run_models=True, positive="event", negative="clear"):
        """Run ``rare-event-eval train`` in this process. Returns (exit code, stdout/json text, stderr, output dir)."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "table.csv"
        source.write_bytes(table_to_csv(columns))
        argv = ["train", str(source), "--target", "label", "--positive-label", positive]
        if negative is not None:
            argv += ["--negative-label", negative]
        argv += ["--model", "logistic", *flags, "--output", str(root / "out.json"), "--html", str(root / "out.html")]
        runner, patch = patched_runner()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if run_models:
                with patch:
                    code = main(argv)
            else:
                code = main(argv)
        output = (root / "out.json")
        return code, (json.loads(output.read_text()) if output.exists() else None), stderr.getvalue(), root

    def test_numeric_only_input_keeps_the_numeric_path(self):
        code, result, _, _ = self.invoke(numeric_table())
        self.assertEqual(code, 0)
        self.assertNotIn("adapter", result["preprocessing"])  # the 0.3.0 numeric parser and recipes, unchanged
        self.assertFalse(result["preprocessing"]["mixed_type_support"])

    def test_id_exclude_and_predictor_flags_are_applied_and_recorded(self):
        columns = numeric_table()
        columns["rowid"] = [str(i) for i in range(len(columns["label"]))]
        code, result, _, _ = self.invoke(columns, "--id-column", "rowid", "--exclude-column", "f3")
        self.assertEqual(code, 0)
        self.assertEqual(result["preprocessing"]["adapter"]["recipe_variant"], "numeric")
        self.assertEqual(result["predictors"], ["f0", "f1", "f2", "constant", "empty"])
        roles = {item["column"]: item["role"] for item in result["excluded_columns"]}
        self.assertEqual((roles["rowid"], roles["f3"], roles["label"]), ("id", "excluded_by_user", "target"))
        code, result, _, _ = self.invoke(columns, "--predictor", "f0", "--predictor", "f1")
        self.assertEqual(code, 0)
        self.assertEqual(result["predictors"], ["f0", "f1"])

    def test_one_vs_rest_flag(self):
        columns = numeric_table()
        columns["label"] = [("event" if value == "event" else ("clear" if i % 2 else "other")) for i, value in enumerate(columns["label"])]
        code, _, stderr, _ = self.invoke(columns, run_models=False)
        self.assertEqual(code, 2)
        self.assertIn("one_vs_rest", stderr)
        code, result, _, _ = self.invoke(columns, "--one-vs-rest", negative=None)
        self.assertEqual(code, 0)
        self.assertEqual(result["label_mapping"]["mode"], "one_vs_rest")
        self.assertEqual(result["label_mapping"]["negative_labels"], ["clear", "other"])

    def test_negative_label_and_one_vs_rest_are_mutually_exclusive(self):
        code, _, stderr, _ = self.invoke(numeric_table(), run_models=False, negative=None)
        self.assertEqual(code, 2)
        self.assertIn("--negative-label", stderr)
        code, _, stderr, _ = self.invoke(numeric_table(), "--one-vs-rest", run_models=False)
        self.assertEqual(code, 2)

    def test_missing_target_policy(self):
        columns = numeric_table()
        columns["label"][4] = ""
        code, result, stderr, _ = self.invoke(columns, run_models=False)
        self.assertEqual(code, 2)  # unchanged numeric-path behavior: a missing label is an error
        self.assertIn("missing label", stderr)
        code, result, _, _ = self.invoke(columns, "--missing-target", "drop_reported")
        self.assertEqual(code, 0)
        self.assertEqual(result["input_parsing"]["rows_excluded"], 1)
        self.assertEqual(result["input_parsing"]["row_exclusions"][0]["row_index"], 4)

    def test_categorical_limits_flag_fails_before_any_model(self):
        code, result, stderr, _ = self.invoke(mixed_table(), "--max-categories", "3", run_models=False, positive="fraud", negative="ok")
        self.assertEqual(code, 2)
        self.assertIn("cardinality", stderr)

    def test_categorical_reject_keeps_the_numeric_error(self):
        code, _, stderr, _ = self.invoke(mixed_table(), "--categorical", "reject", run_models=False, positive="fraud", negative="ok")
        self.assertEqual(code, 2)
        self.assertIn("nonnumeric feature", stderr)

    def test_categorical_input_without_pandas_gives_a_clear_error(self):
        with mock.patch("rare_event_eval.mixed._import_pandas", side_effect=ImportError("no pandas")):
            code, result, stderr, _ = self.invoke(mixed_table(), run_models=False, positive="fraud", negative="ok")
        self.assertEqual(code, 2)
        self.assertIsNone(result)
        self.assertIn("pandas", stderr)
        self.assertIn("[app]", stderr)

    @unittest.skipUnless(pandas, "pandas is absent")
    def test_categorical_input_uses_the_adapter_automatically(self):
        code, result, _, root = self.invoke(mixed_table(), "--data-description", "Fictional mixed table.",
                                            positive="fraud", negative="ok")
        self.assertEqual(code, 0)
        row = result["outcomes"]["logistic"]
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["execution"]["recipe_variant"], "mixed")
        self.assertEqual(result["preprocessing"]["adapter"]["recipe_variant"], "mixed")
        self.assertEqual(result["predictor_kinds"]["city"], "categorical")
        self.assertTrue(row["report"]["data"]["description"].startswith("Fictional mixed table."))
        self.assertIn("Fictional mixed table.", (root / "out.html").read_text())
        for key in ("test_truth", "test_scores", "test_indices", "test_groups"):
            self.assertNotIn(key, row)

    @unittest.skipUnless(pandas, "pandas is absent")
    def test_group_split_flags_work_with_categorical_input(self):
        columns = mixed_table(n=160, positives=40, with_group=True)
        code, result, _, _ = self.invoke(columns, "--split", "group", "--group-column", "entity", positive="fraud", negative="ok")
        self.assertEqual(code, 0)
        self.assertEqual(result["split"]["requested"]["mode"], "group")
        self.assertNotIn("entity", result["predictors"])


if __name__ == "__main__":
    unittest.main()
