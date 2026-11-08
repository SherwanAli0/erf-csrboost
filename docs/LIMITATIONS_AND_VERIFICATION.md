# Limitations and verification

<!-- november-2026-results -->
The latest result record is the [November 2026 aligned extension](https://github.com/SherwanAli0/erf-csrboost/tree/main/research/results/november2026_aligned_extension): 2,000 new completed fits over five added datasets. Over 20 datasets ERF has higher mean AP than reimplemented CSRBoost on 16 (lower on 4) and than a 100-tree plain random forest on 12 (tied on 2, lower on 6; mean AP difference -0.0008, neither an advantage nor equivalence). Claims, limitations and related work: [RESEARCH-EVIDENCE.md](../research/RESEARCH-EVIDENCE.md). Earlier sections below describe historical checks at their recorded dates and do not include the new experiment.
<!-- /november-2026-results -->

What this project does not do or does not establish, what was checked and how, and what was not
checked. Nothing here is a claim of correctness beyond the checks listed.

ERF-CSRBoost is an experimental extension of CSRBoost and a research prototype that is not peer reviewed. The
web application is a demonstration, not a production service.

Status labels: **newly executed** (run during release preparation and the output was read),
**read from code** (a person read the code), **review check** (run once by a review pass at an
earlier commit with scratch scripts that are not in this repository, so a reader cannot repeat that
exact check from the repository), **not verified**.

## 1. Current limitations

### Research results

- The result tables (the June four-way record) are a historical, exploratory comparison on 15
  development datasets with a fixed 0.5 threshold and untuned baselines. They are not an independent
  evaluation. See `RESULTS_EVIDENCE.md`, sections 3.4 and 7, for the full list.
- ERF-CSRBoost uses 1,000 trees against 50 stumps (CSRBoost) and 100 trees (random forest
  baselines). No equal-capacity comparison was run on the 15 datasets.
- On the 15 datasets ERF-CSRBoost is not distinguishable from a balanced random forest on AUC, and
  its mean G-Mean at 0.5 is lower than both CSRBoost and RF-BRF.
- Several datasets contain duplicate rows (ESDRP, FLARE-F), CARGOOD and CARVGOOD share features,
  YEAST5 and YEAST5ERL share their whole feature set, and YEAST5ERL has 5 positives. The CB positive
  class was corrected from the larger to the smaller class in the June record; the April run, kept as
  the earlier run, used the larger class. Data quality is described in more detail below.
- The Python and library versions of the stored runs were not recorded. The June ESR and DCCC rows
  agree with the 25 April runs and not with the 28 April run, for a reason that was not established.
  Bit-for-bit reproduction is not promised.
- The extended exploratory run on 18 datasets (`RESULTS_EVIDENCE.md`, section 8) is uncorrected for
  multiple comparisons, uses a different protocol (models trained on 70 percent of each training
  fold), includes five external datasets chosen by the author without a recorded rule, has gains that
  concentrate in a few datasets, and cannot be reproduced from this repository because its script is
  not included. Beyond its comparison with a 1,000-tree balanced random forest, which shows no AUC or
  AP advantage for ERF-CSRBoost, no equal-capacity result exists.
- The CSRBoost used for comparison is a local re-implementation, not the authors' code, and ERF's
  clustering step (MiniBatchKMeans, no class-redefinition guard) is "CSRBoost-style", not identical
  to the published step.

### Data quality and runner behaviour in the research record

These points concern the stored June record and the runner. None of them changes a number: they
limit what the numbers can be taken to show. Full tables and the status of each figure are in
`RESULTS_EVIDENCE.md`, section 3.5.

- **Exact feature twins between training and test rows.** Many test rows have an identical feature
  vector in the training rows. The counts below are for the first fold only (the first split of the
  runner's `RepeatedStratifiedKFold`: 5 splits, 20 repeats, `random_state` 42). They are a
  one-split diagnostic, not an average over the 100 evaluations. A twin is counted whatever its
  class label.

  | Dataset | Test rows with a feature twin in training | Test rows in that fold |
  |---|---:|---:|
  | ESDRP | 79 | 104 |
  | FLARE-F | 177 | 214 |
  | DCCC | 20 | 6,000 |
  | ILPD | 4 | 117 |
  | YEAST5 | 7 | 297 |
  | YEAST5ERL | 7 | 297 |

  FLARE-F has 19 feature patterns that occur with both classes, and DCCC has 21. All of these figures
  were recomputed during release preparation (loaders only, no model fitted; the scratch script is
  not in this repository), so their status is "newly executed".
- **Twins are not resampling before splitting.** The runner fits the scaler on the training fold and
  gives only training rows to clustering and SMOTE (read from code), so no learned step sees a test
  row. That is a separate matter from the original data containing train/test feature twins, which
  remain whatever the pipeline does. The twins can inflate test scores, and the share of affected test
  rows is largest on ESDRP and FLARE-F. The saved results were not deduplicated, and no deduplicated or group-aware rerun exists.
- **Datasets that share a feature collection.** CARGOOD and CARVGOOD hold the same 1,728 feature rows
  (six features), and YEAST5 and YEAST5ERL hold the same 1,484 feature rows (eight features); in each
  pair only the target label differs. The 15 dataset labels are therefore not 15 separate feature
  collections, while the dataset-level tests (Wilcoxon, Friedman) still treat the datasets as
  independent.
- **SMOTE failure is silent.** The runner's `_apply_smote` (lines 448 to 455 of
  `research/scripts/comparison_rf_csrboost_erf.py`) catches any exception and returns its input, so a
  resampling failure becomes a silent fallback to the retained rows without oversampling. It is
  called for CSRBoost and for each ERF bag. The run keeps no per-fold fallback record, so it was not
  established whether a fallback occurred in the stored June record.
- **A finished run is not proof of a complete experiment.** A dataset whose loader fails is printed
  as `SKIP -- load error` and skipped (lines 716 to 718); a method that raises in a fold is stored as
  NaN (lines 740 to 742) and left out of the summary means. Count the rows and look for NaN after a
  run. The stored June record is complete: 6,000 rows, 15 datasets x 4 methods x 100 folds, no NaN
  (recomputed).
- **GLASS file version.** The selected file has 196 rows (type 3 absent, type 1 has 69 rows). The
  canonical UCI file has 214 rows (type 1 has 70, type 3 has 17). A rerun on the 214-row file would
  not be the same experiment: record the file hash and the subset definition when comparing GLASS
  results.

### The web application

- The application's ERF is adapted from the research runner and is not numerically identical to it
  (`IMPLEMENTATION_MAP.md`). Its results do not reproduce the research tables. It uses a single
  75/25 split by default, which is a demo-scale comparison and not the research protocol.
- The comparison panel is illustrative. One split, no uncertainty estimate. The threshold slider is
  exploratory: it uses the labelled held-out rows, while the headline metrics and the verdict stay
  at 0.50. One threshold is applied to three differently calibrated models.
- Models are trained on at most 4,000 stratified training rows in the single-split comparison
  (fold runs cap training rows at 8,000, 4,000 and 1,500 for 3, 20 and 100 folds). The held-out set
  is never capped. The "100 folds" option on user data is not the research protocol.
- Compared models have unequal capacity: the plain baseline is one 200-tree forest with class
  weights, ERF is 1,000 trees, and the CSRBoost comparator is AdaBoost of 50 default trees. The
  CSRBoost comparator in the comparison (`backend/models/csrboost_exact.py`: KMeans with
  `n_init=10`, the class-redefinition guard, and plain SMOTE on the preprocessed, one-hot matrix) is
  not the same code path as the CSRBoost front end in the balanced download and the ERF bags
  (MiniBatchKMeans with `n_init=1`, no guard, and SMOTE-NC when there are categorical columns). See
  `IMPLEMENTATION_MAP.md`.
- On an exact tie in class counts, the positive class is the alphabetically later label. This rule
  is deterministic but not shown in the UI.
- The generated "leakage-safe pipeline" snippet is an illustration, not the exact application code.
  It assumes the data frame `df` is already loaded, a 0/1 target where 1 is the rare class, and
  numeric, NaN-free features (the snippet says so). It silently drops categorical columns.
- One fixed seed (42) is used. A single fixed-seed result does not show statistical stability. Other
  platforms, CPU counts, library versions and seeds were not tested.

### Data and samples

- The bundled sample files are modified copies of UCI datasets (`../samples/README.md`,
  `../THIRD_PARTY_NOTICES.md`). The credit sample's row selection cannot be reproduced.
- **Sample card text (fixed).** The in-app card text for two samples used to describe the wrong
  class. In `liver_disease.csv` the rare class (target 1, 167 rows) is the group WITHOUT liver
  disease, but the card said "spot patients with liver disease". In `student_dropout.csv` the rare
  class (target 1, 794 rows) is "Enrolled", but the card said "flag students at risk of dropping
  out". The sample files and their class counts were always correct. The cards in
  `backend/server.py` now read "find the less common group: records without liver disease" and
  "find the less common group: currently enrolled students", the page's demo noun for the liver
  sample is "records" instead of "patients", and the detector wording says "likely rare cases"
  instead of "high-risk". `backend/test_sample_cards.py` checks the rare class, row count and ratio
  of all four samples against the card text. The credit and car cards were checked against the data
  (1,327 defaulters, 69 "good" cars) and were already correct.
- No bundled sample has categorical columns, so the SMOTE-NC path is not exercised by a sample.
- Rights for the ESR and KEEL data files are not established (`../THIRD_PARTY_NOTICES.md`), so they
  are not redistributed.

### Privacy and server state

See section 6 of `../README.md`. In short: uploaded file contents are processed in memory and not
saved; a pseudonymous usage event is stored; there is no retention limit or purge job. Platform-level
logs of the hosting provider are outside the application and were not inspected.

## 2. What was checked, with commands and results

All commands run from the repository root with Python 3.11.9 in a fresh virtual environment built
from `requirements.txt` (pandas 2.3.3, numpy 2.2.6, scikit-learn 1.8.0, imbalanced-learn 0.14.1,
openpyxl 3.1.5, xlrd 2.0.2; `pip check` reported no broken requirements in that environment).
Checked on Windows 11. Linux and macOS were not tested.

| Check | Command | Result | Status |
|---|---|---|---|
| Server tests | `python backend/test_head.py` | 2 tests, OK | Newly executed |
| Messy-input suite | `python backend/test_torture.py` | 48 passed, 0 failed of 48 | Newly executed (final tree of this release branch). The suite generates its inputs inside the file and checks that each realistic messy file either processes or fails with a clear message. It does not check the metric values |
| Rare-class split guard | `python backend/test_rare_split_guard.py` | 2 tests, OK. The test fails on the code before the fix (a 2-row rare class gave a "successful" comparison with no rare rows in the test set) | Newly executed on this branch; the failure on the earlier code was shown when the test was written |
| Small fixes | `python backend/test_small_fixes.py` | 3 tests, OK. They cover the "other" label collision, the reported fold downgrade, and the snippet assumptions. The tests fail on the earlier code | Newly executed on this branch |
| Verdict logic | `node frontend/test_verdict.js` | 13 cases, all pass. Every case also checks that the results nav chip (ahead, behind, mixed, about the same, balanced) agrees with the verdict box, so a tie cannot show as "ahead". The first 7 cases failed against the earlier `index.html` (the earlier verdict could call ERF-CSRBoost equal or ahead when it was clearly worse); the chip cases were added with the chip fix | Newly executed |
| Receipt and notes text | `node frontend/test_receipt.js` | 13 checks pass: single-split and fold-run receipt wording, the capacity note, no long dashes, the slider label. There is no DOM test for the subsample note or the fold-downgrade sentence | Newly executed |
| Sample card text | `python backend/test_sample_cards.py` | 3 tests, OK. For each bundled sample the class the app picks as positive is target 1 with the expected size (167, 1,327, 794, 69 rows), the row count and ratio in the card text match the file, and the liver and student cards name the right class. The test fails on the earlier card text and page noun | Newly executed |
| Landing-page preview numbers | See "Landing-page preview numbers" below | Reproduced | Newly executed |
| Local server | `PORT=8631 python backend/server.py`, then `GET /healthz`, `GET /`, `HEAD /`, `GET /api/samples` | 200, 200 (the page body of the committed `frontend/index.html`, 66,506 bytes), 200, sample list returned | Newly executed with the environment's interpreter. The literal `py -3.11 backend/server.py` (Windows launcher) was not run against a global interpreter |
| Live app | `GET https://fixyourdataset.fly.dev/healthz` | 200 | Newly executed (one read-only request) |
| Feature twins, shared feature sets, GLASS counts | An isolated scratch script that copies the runner's loader logic (the runner module was not imported or run), loads the data files, builds the first split of the runner's splitter and compares feature rows in memory | First-fold twin counts of the six datasets above, 19 and 21 conflicting feature patterns, identical feature rows in CARGOOD and CARVGOOD and in YEAST5 and YEAST5ERL, 196 GLASS rows with the class counts given above, complete June record (6,000 rows, no NaN). Details: `RESULTS_EVIDENCE.md`, section 3.5 | Newly executed, with a local Python 3.11 environment rather than the fresh virtual environment above. No model was fitted. The scratch script is not in this repository |
| Datasets and CSV | recomputation of the result tables from the June four-way record and of the exploratory run's tables, sample label counts, sample row matching | See `RESULTS_EVIDENCE.md` and `../samples/README.md` | Newly executed |

### Landing-page preview numbers

The landing page preview card ("A real result from the liver demo") shows a leaky balanced accuracy of
0.741, an honest one of 0.701, and "+0.040". These were checked by exporting the branch tip with
`git archive`, starting `backend/server.py` from the export on a free local port, and making the same
requests as the page's demo button: `GET /api/sample?name=liver_disease.csv`, `POST /api/profile`,
then `POST /api/analyze` with the suggested target, method `clustered_smote`, classifier `erf`,
`folds` 1 (single split, seed 42) and `do_classify` true. The response's `leakage_lab` held leaky
balanced accuracy 0.7408, honest 0.7005 and a delta of 0.0403, which the page shows as 0.741, 0.701
and +0.040. Two runs in the same server process gave identical values. The Leakage Lab uses a plain
random forest for both arms, so these numbers are not the ERF-CSRBoost scores. Limits of this check:
one machine (Windows 11, Python 3.11.9, pandas 2.3.3, scikit-learn 1.8.0, imbalanced-learn 0.14.1),
numpy 2.4.6 rather than the pinned 2.2.6, one seed, and the figures describe the single split of the
liver demo only, not a general effect size.

`test_head.py` replaces the usage logger with a stub; neither test reads personal uploads or
downloads data. Importing the server creates a local `data_local/` folder (ignored by Git).

What the existing suites do not cover: which class the metrics describe, whether preprocessing is
fitted on training rows only, determinism, tiny-minority outcomes, the verdict text, the snippet and
the storage behaviour. The boundary checks in section 3 came from separate scripts.

## 3. Boundary checks

These were run by a review pass at commit `6aa0778` (the code before this branch's fixes), using
scratch scripts that are not in this repository. They were not re-run on the final commit of this
branch. The fixes on this branch touch only the items marked "fixed" below.

| Boundary | Result at `6aa0778` | Status on this branch |
|---|---|---|
| Learned preprocessing (imputer, scaler, one-hot, ordinal encoders, KMeans) fitted on training rows only, single split | Holds. Instrumented with test-only sentinel values: no learned step saw any test row | Review check; not re-run |
| Resampling excludes held-out rows | Holds: each resampling call received only the 600 training rows of an 800-row file, for the plain, CSRBoost and all 20 ERF bags | Review check; not re-run |
| Same, fold mode | Holds for ERF (train and test index sets disjoint, every row held out once); plain and CSRBoost use the same per-fold arrays (code) | Review check and code reading |
| Deliberate exceptions to train-only fitting | The Leakage Lab's leaky arm resamples all rows on purpose. "Score your own file" trains on all labelled rows including held-out ones. Leakage scan and the ID-like column filter read whole-file features or counts for diagnostics only. None of these feeds the reported test metrics | Read from code and review check |
| Positive class is the rare class, also when it sorts first, for string labels, and with numeric labels | Holds in 7 scenarios. Recall, precision, PR-AUC and ROC-AUC recomputed independently from the probability payload agree to within 3e-4 | Review check; not re-run |
| Probability column alignment across ERF bags | Holds: all 20 bags have `classes_` equal to [0, 1] and the reported probability is column 1. AUC against the rare flag was 0.83 to 0.99 | Review check; not re-run |
| Multiclass target | Collapsed to rarest class against the rest, with a warning. A rarest class literally named "other" gave a raw error | Name collision fixed (regression test) |
| Minority of 1 row | Clear error message | Review check |
| Minority of 2 rows | Defect: the split could leave no rare row in the test set and the run reported success with meaningless 1.000 scores | Fixed: the comparison now refuses with a clear message (regression test) |
| Minority of 3 to 6 rows | Runs, with a "small dataset, illustrative and noisy" warning; where SMOTE cannot run the engine falls back to random oversampling and says so | Review check |
| Fold run with fewer rare rows than splits | Silently fell back to one split | Fixed: `folds_downgraded_from` is set (backend test) and the UI states the downgrade (code inspection only; not viewed in a browser) |
| Unsupported inputs (single class, target-only file, regression-like target, ID-like only, 25 classes, empty table, missing target column) | Clear messages | Review check; also exercised by the 48-case suite |
| A request with a missing field; a negative `Content-Length` | Raw `'csv'` error text; a negative length made the server wait for the client to close | Fixed: readable 400 error, and a negative length is refused (checked on a local server over a raw socket) |
| Four-row file | Raw scikit-learn message about `test_size` | Not fixed |
| Plain, CSRBoost and ERF scored on identical held-out rows | Holds in the single split and in fold mode. The test set is never capped | Review check; not re-run |
| Fold-mode receipt and locked-test-set text | The text said scores were measured on the downloadable 25 percent rows even though fold scores are pooled out-of-fold predictions | Fixed (receipt test) |
| Threshold slider | Was not labelled exploratory | Fixed: labelled exploratory with a note (receipt test). Not viewed in a browser |
| Slider payload on a large test set | The payload is subsampled to 5,000 rows, so slider counts differ slightly from the metrics table | A note is shown when the payload is subsampled (code inspection only; no test; not viewed in a browser) |
| Repeated fixed-seed runs | Identical across two runs in one process and in a separate process with `OMP_NUM_THREADS=1` on the liver sample | Review check; not re-run. Says nothing about other platforms or seeds |
| Verdict headline | Could call ERF equal or ahead when it was clearly worse | Fixed (node harness, 13 cases including the nav chip) |
| Generated snippet | Failed for string labels or NaN features and silently clustered the wrong class for a 1-is-majority target | Assumptions now stated in the snippet; the snippet was not re-executed after that comment-only change |

## 4. Not verified

- Live deployment behaviour beyond one `/healthz` request, every `fly` command in `DEPLOY_FLY.md`,
  and whether the `/data` volume is writable by the non-root container user. The usage logger
  swallows write errors, so a failed write would not be visible except as an empty `/stats` page.
- Whether `STATS_KEY` is set on the live app. Without it the visitor tag uses a constant key and is
  weaker pseudonymisation.
- Browser rendering of any screen. The verdict function, the receipt text and the script syntax were
  run in node; the slider note, the subsample note and the locked-test bar were not viewed or clicked
  in a real browser.
- Memory and timing for uploads near the 30 MB limit on the 1 GB machine.
- The documented research command (`research/scripts/comparison_rf_csrboost_erf.py`) was read from
  code but was not executed during release preparation, and no training was run. The 10-hour full run
  and the smaller invocations in `../README.md` are untested here.
- Whether a SMOTE fallback (`_apply_smote`, lines 448 to 455 of the runner) happened in the stored
  June record. No per-fold record exists. Likewise, the effect of the feature twins on any stored
  number: no deduplicated or group-aware evaluation was run.
- Feature twins and conflicting-label patterns of the five datasets added in the exploratory run, and
  of the datasets not listed in the first-fold table (PSDAS, ESR, BCW, CB, SEED, WINE, GLASS, CARGOOD,
  CARVGOOD).
- Reproduction of the exploratory 18-dataset run: not possible from this repository (script, data
  and library versions are not included), and the stored script is not shown to be the exact text that
  produced every dataset.
- The statement that the June ESR and DCCC reruns reproduce the 25 April runs. It comes from the
  author's own records, which are not part of this repository.
- That the application reproduces any research number. It does not (`IMPLEMENTATION_MAP.md`).
- Python and library versions of the stored runs; other seeds, platforms and library versions.
- The CSRBoost paper's own tables were not re-checked against the 15 datasets used here.
- The origin of the two inline SVG icons in `frontend/index.html`.
- Every dataset's redistribution terms beyond what the provider pages stated on 4 November 2026
  (ESR could not be read).

## 5. Repeat the checks

Run these from the repository root with the interpreter of the virtual environment built from
`requirements.txt` (README section 3). Windows is shown; on macOS or Linux use `.venv/bin/python`.
If you activate the environment first, `python` is that interpreter and you can use it instead of
the path. The two `node` tests need Node 18 or newer and no packages.

```
.venv\Scripts\python backend/test_head.py
.venv\Scripts\python backend/test_rare_split_guard.py
.venv\Scripts\python backend/test_small_fixes.py
.venv\Scripts\python backend/test_sample_cards.py
.venv\Scripts\python backend/test_torture.py          # takes several minutes
node frontend/test_verdict.js
node frontend/test_receipt.js
```

The tests use synthetic data generated inside the files and the four bundled samples. Do not point
them at personal data.

### Network exposure of the local server

`python backend/server.py` listens on all network interfaces (`0.0.0.0`), not only on `localhost`,
and has no login. While it runs, other devices on the same network can reach it at your machine's
address, and anything uploaded to it is processed in that process's memory. The listening address
is fixed in the code and there is no setting for it; this release does not change the code. To keep
the server off the network, use a firewall rule for the port or run the Docker image with the port
published to the loopback address only (`docker run -p 127.0.0.1:8600:7860 <image>`). The Docker
route was not tested here. The Fly.io deployment is meant to be public.
