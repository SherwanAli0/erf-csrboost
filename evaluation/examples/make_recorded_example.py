"""Regenerate the recorded example: a small app run on a seeded synthetic table (provenance: recorded app run).

The table is generated here from a seeded random number generator. It contains no real person, no external data
and nothing confidential. The comparison is run through the application's own job path (the job coordinator that
backs ``backend/api.py``: a dataset upload, a job in a worker process, the stored ``training-1.1`` report and the
HTML export), so the saved files are what the app produces, not hand-written. Nothing is edited afterwards except the
documented additive fields ``provenance``, ``app`` and ``example``.

Run it from the repository root with the application's environment (the evaluation package, pandas,
imbalanced-learn, FastAPI and its dependencies installed)::

    python evaluation/examples/make_recorded_example.py

That writes ``recorded_example_data.csv`` and ``output/recorded-example.json`` and ``.html`` next to this file, and
the same two report files as ``recorded-example.json`` and ``recorded-example.html`` in the package's ``web`` folder.
``--into DIR`` writes the three files into one folder instead (used by the test that checks regeneration). A run
takes about half a minute. The fits are deterministic for the same versions; the creation time and the timings in the
report differ from run to run, and every aggregate metric stays the same.

The set of models and the settings are fixed below. The generator's signal is additive by construction (a linear
log-odds), which suits a linear model; that is stated in the example's own note and is not evidence about other data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WEB_DIR = ROOT / "evaluation" / "src" / "rare_event_eval" / "web"

DATA_SEED = 6
ROWS = 1500
JOB_PARAMS = {
    "target": "outcome", "positive_label": "default", "negative_label": "repaid", "id_columns": ["customer_id"],
    "models": ["logistic", "rf_weighted", "balanced_rf", "hist_gradient_boosting", "erf"],
    "split_mode": "stratified", "seed": 7, "calibration": "none", "review_fractions": [0.05, 0.1, 0.2],
    "costs": {"false_positive_cost": 1.0, "false_negative_cost": 10.0, "review_cost": 0.1},
}
DATA_FILE = "recorded_example_data.csv"
WAIT_SECONDS = 900
COMMAND = "python evaluation/examples/make_recorded_example.py"
TITLE = "Synthetic loan outcomes, 1,500 rows, five models"


def synthetic_csv(seed: int = DATA_SEED, rows: int = ROWS) -> bytes:
    """Seeded loan-style table. ``outcome`` is ``default`` (rare) or ``repaid``; ``customer_id`` is an identifier and
    ``noise_a`` and ``noise_b`` carry no signal. A few cells are blank on purpose."""
    rng = np.random.RandomState(seed)
    age = np.clip(rng.normal(41, 12, rows), 19, 80)
    income = np.exp(rng.normal(10.6, 0.45, rows))
    debt = rng.beta(2.2, 4.0, rows)
    tenure = rng.poisson(48, rows)
    region = rng.choice(["north", "south", "east", "west", "central"], rows, p=[.25, .2, .2, .2, .15])
    employment = rng.choice(["salaried", "self_employed", "contract"], rows, p=[.6, .2, .2])
    noise_a = rng.normal(0, 1, rows)
    noise_b = rng.normal(0, 1, rows)

    def z(values):
        return (values - values.mean()) / values.std()

    logit = (-2.65 + 1.15 * z(debt) - 0.7 * z(np.log(income)) - 0.45 * z(age) - 0.35 * z(tenure)
             + 0.5 * (employment == "contract") + 0.4 * (region == "south"))
    outcome = rng.rand(rows) < 1 / (1 + np.exp(-logit))
    lines = ["customer_id,age,income,debt_ratio,tenure_months,region,employment,noise_a,noise_b,outcome"]
    for i in range(rows):
        lines.append(",".join([
            f"C{i + 1:05d}", "" if (i * 7 + 1) % 41 == 0 else f"{age[i]:.1f}", f"{income[i]:.0f}", f"{debt[i]:.3f}",
            str(tenure[i]), region[i], "" if (i * 7 + 3) % 41 == 0 else employment[i], f"{noise_a[i]:.3f}",
            f"{noise_b[i]:.3f}", "default" if outcome[i] else "repaid"]))
    return ("\n".join(lines) + "\n").encode("ascii")


def _use_backend():
    for entry in (str(ROOT / "backend"), str(ROOT / "evaluation" / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def run_app_job(raw: bytes) -> tuple:
    """Run the comparison through the coordinator (the app's job path). Returns ``(report, app_context)``."""
    _use_backend()
    import api
    from jobs import Coordinator, JobConfig, states
    app = {"app_version": api.APP_VERSION, "release_id": api._release_id()}
    root = Path(tempfile.mkdtemp(prefix="fyd-recorded-example-"))
    coordinator = Coordinator(JobConfig(temp_root=str(root / "storage")))
    coordinator.app_context = app
    coordinator.start()
    try:
        dataset = coordinator.create_dataset(raw, DATA_FILE)
        job = coordinator.submit_job(dataset["dataset_id"], dataset["capability"], dict(JOB_PARAMS))
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            status = coordinator.job_status(job["job_id"], job["capability"])
            if status["state"] not in states.LIVE_STATES:
                break
            if time.monotonic() > deadline:
                raise SystemExit("The comparison did not finish in time.")
            time.sleep(0.5)
        if status["state"] != states.SUCCEEDED:
            raise SystemExit(f"The comparison ended in state {status['state']} (every model must succeed): "
                             f"{status.get('reason')}")
        _, report = coordinator.job_result(job["job_id"], job["capability"])
        return report, app
    finally:
        coordinator.stop()
        shutil.rmtree(root, ignore_errors=True)


def build_example_block(report: dict, services) -> dict:
    """Plain facts about this example, computed from the report so they cannot disagree with it."""
    parts = report["split"]["achieved"]["partitions"]
    test = parts["test"]
    ap = {name: row["report"]["models"][name]["metrics"]["average_precision"] for name, row in report["outcomes"].items()}
    ranked = sorted(ap.items(), key=lambda item: -item[1])
    best, best_ap = ranked[0]
    simple_ap = ap["logistic"]
    others = [value for name, value in ap.items() if name != "logistic"]
    if simple_ap >= max(others):
        verdict = (f"The simplest model, logistic regression, has the highest average precision here ({simple_ap:.3f}); "
                   f"the best of the other four is {max(others):.3f}.")
    elif simple_ap >= best_ap - 0.05:
        verdict = (f"The simplest model, logistic regression, is competitive: average precision {simple_ap:.3f} against "
                   f"{best_ap:.3f} for {best}, the highest.")
    else:
        verdict = (f"Logistic regression reaches average precision {simple_ap:.3f}; {best} is higher at {best_ap:.3f}.")
    replay_argv, _ = services.replay_argv(
        services.effective_settings(JOB_PARAMS, {"max_expanded_features": 100}, 60.0), f"evaluation/examples/{DATA_FILE}")
    return {
        "title": TITLE,
        "description": (f"A seeded synthetic table of {ROWS:,} loan-style rows with a rare outcome (default, "
                        f"{report['label_mapping']['counts']['positive_rows']} rows). Five models were compared on one "
                        f"stratified split with the settings the app uses for an uploaded file. {verdict}"),
        "data_origin": (f"Generated by evaluation/examples/make_recorded_example.py from a seeded random number "
                        f"generator (seed {DATA_SEED}). It contains no real person, no external data and nothing "
                        f"confidential."),
        "note": (f"The signal in this table is additive by construction, which suits a linear model, so this example "
                 f"says nothing about which model suits other data. The test partition has {test['positive_rows']} "
                 f"positive rows; small differences between models should not be read as a ranking. Costs are "
                 f"illustrative assumptions in arbitrary units, not measured savings."),
        "generator": {"script": "evaluation/examples/make_recorded_example.py", "data_seed": DATA_SEED, "rows": ROWS,
                      "data_file": f"evaluation/examples/{DATA_FILE}", "regenerate_command": COMMAND,
                      "replay_command_arguments": replay_argv},
        "average_precision_on_test": {name: round(value, 6) for name, value in ap.items()},
    }


def generate() -> tuple:
    """``(csv bytes, report json bytes, report html text)`` for the recorded example."""
    _use_backend()
    import services
    raw = synthetic_csv()
    report, app = run_app_job(raw)
    recorded = services.annotate_report(report, services.RECORDED_RUN, app)
    recorded["example"] = build_example_block(recorded, services)
    return raw, services.dump_json(recorded), services.render_report_html(recorded)


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--into", type=Path, help="write the three files into this folder only")
    args = parser.parse_args(argv)
    raw, report_json, report_html = generate()
    html_bytes = report_html.encode("utf-8")
    if args.into:
        targets = [(args.into / DATA_FILE, raw), (args.into / "recorded-example.json", report_json),
                   (args.into / "recorded-example.html", html_bytes)]
    else:
        targets = [(HERE / DATA_FILE, raw), (HERE / "output" / "recorded-example.json", report_json),
                   (HERE / "output" / "recorded-example.html", html_bytes),
                   (WEB_DIR / "recorded-example.json", report_json), (WEB_DIR / "recorded-example.html", html_bytes)]
    for path, data in targets:
        write_new(path, data)
        print(f"wrote {path.name} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
