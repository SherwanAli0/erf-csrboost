"""Shared fixtures for the mixed-type adapter tests: small inline tables and an in-process worker runner."""
from datetime import date, timedelta
import json
from pathlib import Path
import re
import sys
from unittest import mock

import numpy as np

from rare_event_eval import worker


def mixed_table(n=140, seed=3, positives=28, with_row_id=False, labels=("fraud", "ok"), city_values=("a", "b", "c", "d"),
                positive_rows=None, with_time=False, time_ties=3, with_group=False, group_size=4):
    """CSV-like columns (strings; blank means missing) with numeric, categorical and missing cells.

    ``positive_rows`` places the rare class explicitly. ``with_time`` adds ISO dates in ``when`` (``time_ties`` rows
    share each date) and ``with_group`` adds ``entity`` (``group_size`` consecutive rows per group).
    """
    rng = np.random.RandomState(seed)
    y = np.zeros(n, dtype=bool)
    if positive_rows is None:
        y[rng.choice(n, positives, replace=False)] = True
    else:
        y[list(positive_rows)] = True
    age = rng.normal(40, 10, n) + 8 * y
    income = rng.gamma(3, 10, n) - 5 * y
    color = np.where(rng.rand(n) < 0.35 + 0.3 * y, "red", rng.choice(["green", "blue"], n))
    city = rng.choice(list(city_values), n)
    columns = {}
    if with_row_id:
        columns["row_id"] = [str(i) for i in range(n)]
    columns["age"] = ["" if i % 19 == 0 else f"{value:.3f}" for i, value in enumerate(age)]
    columns["income"] = [f"{value:.3f}" for value in income]
    columns["color"] = ["" if i % 23 == 0 else str(value) for i, value in enumerate(color)]
    columns["city"] = [str(value) for value in city]
    if with_time:
        columns["when"] = [(date(2026, 1, 1) + timedelta(days=i // time_ties)).isoformat() for i in range(n)]
    if with_group:
        columns["entity"] = [f"g{i // group_size}" for i in range(n)]
    columns["label"] = [labels[0] if flag else labels[1] for flag in y]
    return columns


def numeric_table(n=150, seed=5, positives=30):
    """Numeric-only columns including missing cells, a constant column and an all-missing column."""
    rng = np.random.RandomState(seed)
    y = np.zeros(n, dtype=bool)
    y[rng.choice(n, positives, replace=False)] = True
    columns = {}
    for index in range(4):
        values = rng.normal(0, 1, n) + (0.8 if index < 2 else 0.0) * y
        columns[f"f{index}"] = ["" if (i + index) % 13 == 0 else f"{value:.6f}" for i, value in enumerate(values)]
    columns["constant"] = ["2.5"] * n
    columns["empty"] = [""] * n
    columns["label"] = ["event" if flag else "clear" for flag in y]
    return columns


def to_arrays(columns, target="label"):
    """Numeric matrix and label array for ``training.train_comparison`` from a numeric table of strings."""
    names = [name for name in columns if name != target]
    X = np.array([[np.nan if value == "" else float(value) for value in columns[name]] for name in names]).T
    return X, np.asarray(columns[target])


def table_to_csv(columns) -> bytes:
    names = list(columns)
    rows = zip(*(columns[name] for name in names))
    return ("\n".join([",".join(names)] + [",".join(row) for row in rows]) + "\n").encode("utf-8")


class InProcessRunner:
    """Stand-in for ``run_worker_directory`` that runs the real worker entry point in this process.

    It records the typed inputs of every call (arrays and config) so tests can inspect them. It is only
    used to avoid a subprocess start per model; a separate test runs the real subprocess.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, directory, model_name, timeout_seconds):
        directory = Path(directory)
        with np.load(directory / "input.npz", allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.calls.append({"model": model_name, "arrays": arrays, "config": config})
        with mock.patch.object(sys, "argv", ["worker", str(directory)]):
            worker.main()
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        result["worker_wall_seconds"] = 0.01
        return result


def patched_runner():
    runner = InProcessRunner()
    return runner, mock.patch("rare_event_eval.mixed.run_worker_directory", runner)


def assert_finite_json(test, value):
    encoded = json.dumps(value, allow_nan=False)
    test.assertNotRegex(encoded, r"\b(NaN|Infinity)\b")
    return encoded


def assert_close(test, first, second, tolerance=1e-12, path="value"):
    """Recursive equality with an absolute tolerance on floats."""
    if isinstance(first, dict):
        test.assertIsInstance(second, dict, path)
        test.assertEqual(sorted(first), sorted(second), path)
        for key in first:
            assert_close(test, first[key], second[key], tolerance, f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second), path)
        for index, (left, right) in enumerate(zip(first, second)):
            assert_close(test, left, right, tolerance, f"{path}[{index}]")
    elif isinstance(first, float) or isinstance(second, float):
        test.assertIsNotNone(first, path)
        test.assertIsNotNone(second, path)
        test.assertLessEqual(abs(first - second), tolerance, f"{path}: {first!r} vs {second!r}")
    else:
        test.assertEqual(first, second, path)


NO_DASHES = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")
