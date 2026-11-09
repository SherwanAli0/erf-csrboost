# Rare-event evaluation workbench

Author: Sherwan Ali. Development release 0.3.0.

Compare binary-classification models using explicit labels, review capacity,
probability quality and assumed costs. Start with saved predictions in the free
static workbench, or install Python to train numeric baselines locally.

The useful output is a traceable evaluation, including cases where a simpler
model is better. ERF is an experimental method within that workflow. The new
144-run exploratory pilot did not meet its predeclared rule for expanding an
ERF-superiority campaign. Historical research results remain separate.

## Try the browser demo

Open `src/rare_event_eval/web/index.html` directly in a browser. The page has five tabs: Overview (with an
"Explore an example" section), Analyze a dataset, Evaluate predictions, Research and How it works. Every result is
labelled as a fictional fixture, a recorded app run, a fresh app run or a historical research record. The fictional
examples have invented scores and are not ERF results. The recorded app run is a real run of this app on a seeded
synthetic table (see "Recorded example" below). Prediction CSV evaluation and aggregate JSON/HTML downloads work in the
browser without Python, an account, AWS, model calls or email, and make no network request. The Research tab holds the
aggregate research results, kept apart by evidence set, with "not peer reviewed" stated.

"Analyze a dataset" depends on how the page is served. Opened from a file or from static hosting it explains that
training needs Python. Served by `python -m rare_event_eval serve` it shows the numeric-feature training form. Served
by the hosted service (`backend/api.py`, see `docs/API.md`) it offers the five-step flow Data, Target and split, Models
and policy, Run, Results and export: the file is uploaded to the service for fitting, progress and Cancel are real, and
the downloads are the report, the run settings and a reproducibility bundle, none with row-level data.

The scoring arithmetic of the browser is `src/rare_event_eval/web/scoring-core.js` (checked against Python fixtures);
the other scripts in that folder are plain files with no build step. Each page script is loaded from the same folder, so
the folder can be copied as a whole.

CSV data stays in the browser in static mode. In local server mode it is sent
only to the loopback service. Training uses temporary local worker files,
removed after the job under normal operation; no uploaded dataset is retained
by the application. Exports contain aggregates, but aggregates can be sensitive.
No analytics or external assets are loaded. Repository links open externally
only when selected.

## Install and run locally

Python 3.11 or newer. The 0.3.0 test suite was run on Windows with Python 3.11.9
(versions in the release record); other interpreters need their own run.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-training-tested.txt ".[training]"
.venv\Scripts\python -m rare_event_eval serve
```

Open http://127.0.0.1:8601. On Linux/macOS use `.venv/bin/python`.
Initial installation needs dependencies locally or network access. Runtime is
offline. For saved predictions only, use `requirements-tested.txt` and `.`.
The package name is a working name, not a published PyPI distribution.

## Evaluate existing predictions

```powershell
.venv\Scripts\python -m rare_event_eval evaluate examples/fictional_predictions.csv --label-column outcome --score-column baseline --score-column candidate --positive-label event --negative-label clear --score-kind probability --threshold 0.5 --review-budget 2 --role test --data-description "Fictional fixture with invented scores." --output reports/example.json --html reports/example.html
```

Labels are exact strings. Higher scores must mean the declared positive class.
UTF-8 CSV, including BOM, is supported. Malformed or missing scores are rejected,
not silently removed. Multiple models share the same evaluation population.
Ranking scores are supported without probability metrics. Existing output files
are never overwritten. The input hash records exact bytes, not the input path.

The Python API remains `evaluate_predictions`, `CostAssumptions` and `render_html`.
See [the report contract](docs/report-contract.md) for metric definitions.
Average precision uses recall-increment weighting, not trapezoidal PR area.
Budget ties use input order and show possible allocations; those bounds are not
confidence intervals. Costs are hypothetical additive assumptions, not savings.

## Train a fresh comparison

```powershell
.venv\Scripts\python -m rare_event_eval train examples/fictional_training.csv --target outcome --positive-label event --negative-label clear --model logistic --model rf_weighted --data-description "Fictional training fixture." --output reports/training.json --html reports/training.html
```

Numeric predictors use the numeric recipes. If a predictor is not numeric, the command switches to the mixed-type
adapter (categorical vocabularies fitted on the training partition, unseen categories as all-zero indicators, SMOTENC for
categorical resampling; see [method notes](docs/methods.md)). That path needs the optional `app` extra (pandas) and reports
a clear error without it; `--categorical reject` refuses such files instead. Further options: `--predictor`,
`--exclude-column`, `--id-column`, `--one-vs-rest` (no `--negative-label`), `--missing-target drop_reported`,
`--max-categories` and `--max-expanded-features`. Failed models are reported with a reason code, never replaced.
The target, optional group column and optional time
column are excluded from predictors. Approximately 60/20/20 train/validation/test
partitions separate preprocessing, threshold selection and final evaluation.
Threshold selection uses 101 fixed validation thresholds, minimizing supplied
costs or maximizing F1, with a highest-threshold tie break. Reports identify fresh
training and explicit model successes, errors and timeouts.

Choose `--split group --group-column entity`, `--split feature_group`, or
`--split chronological --time-column timestamp` when that matches the intended
generalization question. Chronological outer splits preserve timestamp ties.
Combined group/time constraints are unsupported. Duplicate-feature overlap is
reported as a diagnostic, not automatically declared leakage or deleted.

Optional `--calibration sigmoid` fits three-fold calibration within training.
Infeasible folds fail explicitly. Temporal calibration uses ordered inner folds;
equal timestamps can straddle an inner calibration boundary. Use no calibration
when strict separation of all timestamp ties is required. No temporal dataset
was evaluated in the research pilot.

Models: logistic, raw/weighted RF, Balanced RF, histogram gradient boosting,
SMOTE-RF, CSRBoost-style, ERF, a 1,000-tree weighted RF capacity reference and
three ERF ablations. Defaults and limits are in [method notes](docs/methods.md).
No automatic hyperparameter search or model recommendation is performed.

## Export a free static site

```powershell
.venv\Scripts\python -m rare_event_eval export-demo --output-directory exported-workbench
```

The target must be new. Open its index.html directly or review it for static
hosting. The export does not publish anything. A static site evaluates saved
predictions but cannot train Python models.

## Recorded example

`examples/output/recorded-example.json` and `.html` are a recorded run of the hosted application's job path on a seeded
synthetic table (`examples/recorded_example_data.csv`, no real person and no external data), five models, provenance
`recorded app run`. It is not a historical ERF result. `examples/make_recorded_example.py` regenerates it from the
repository root (it needs the application environment: pandas, imbalanced-learn and the FastAPI requirements of
`backend/`):

```powershell
python evaluation/examples/make_recorded_example.py
```

The table and every aggregate metric are reproduced exactly; the creation time and the timings differ per run. The
files are written next to the script and as `recorded-example.json` and `.html` in the package's `web` folder.

## Verify and build

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m pip install build
.venv\Scripts\python -m build
```

Node.js (18 or newer, no packages) enables the browser/Python metric parity check and the browser module tests
(`tests/test_web_*.js`, run through `tests/test_web_node.py`). Without Node those tests are skipped. See [scope](docs/scope-and-decisions.md). No external user trial
has occurred. The `examples/output` reports were recorded with version 0.2.0 and
have not been regenerated.

Code is under the [MIT license](LICENSE). Third-party methods and dependencies
retain their attribution and licenses. Do not bulk-publish the private workspace.
