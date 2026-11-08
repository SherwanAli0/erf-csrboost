# ERF-CSRBoost: Resampled Forest Ensembles for Imbalanced Classification

ERF-CSRBoost combines clustered undersampling, SMOTE and bagged random forests to study classification with imbalanced labels. Its implementation averages the predictions of 20 resampled forests, each containing 50 trees.

This repository contains the research implementation, recorded evaluations and **FixYourDataset**, a web application for comparing models and examining the effects of resampling and data leakage.

**Highlights** (November 2026 evaluation of ERF-CSRBoost, "ERF" below; 20 datasets; mean average precision (AP) per dataset over 100 folds; counts of datasets):

- ERF exceeds the reimplemented CSRBoost on 16 of 20 datasets (+0.0685 mean AP; the largest per-dataset gain is +0.3548 on CARGOOD and the largest loss is -0.0579 on FLAREF) and a 100-tree balanced random forest on 16 of 20 datasets (+0.0093 mean AP).
- ERF is within 0.001 mean AP of a 100-tree plain random forest (0.7872 against 0.7880; ERF higher on 12 datasets, tied on 2, lower on 6), and it has the highest mean AP of the four methods on 7 datasets and is tied for highest on 2 more.
- All 2,000 new fits (five datasets, four methods, 100 folds each) completed with zero failures, all five metrics of every fit were recomputed from the saved predictions within a tolerance of 1e-12 (done on private prediction files), and every table of the November record is covered by a SHA-256 manifest and a read-only checker (`verify_record.py`).
- ERF's own controls are reported next to the headline. In a tree-count-matched run against a 1,000-tree balanced random forest (a different protocol: each method trains on 70% of each outer training fold) the mean AP difference was +0.0010 over 18 datasets (ERF higher on 12, lower on 6; bootstrap 95% interval -0.0171 to +0.0180).
- In a separate 144-cell controlled pilot (approximately 60/20/20 train, validation and test partitions and newer evaluation code, so a different protocol again) ERF had higher mean AP than a balanced random forest on 4 of 5 dataset families (the fifth, WINE, tied). The pilot's predeclared confirmation gate required at least 0.01 AP over the balanced forest on 4 families and a median paired ERF-to-balanced-forest fit-time ratio of at most 3; ERF met the AP condition on 2 of 5 families and the observed time ratio was 5.97, so the gate was not met. A plain random forest had higher mean AP than ERF on 3 of 5 families (SONAR, GLASS and YEAST5), and most of ERF's advantage over the balanced forest comes from YEAST5 (+0.0891, against an average of +0.0049 on the other four families). The pilot's cell records are private (aggregates in `research/RESEARCH-EVIDENCE.md`).
- A separate audit by the same author, independent of the CSRBoost authors, of 143 published CSRBoost comparisons ([csrboost-audit](https://github.com/SherwanAli0/csrboost-audit)) records a result and an average error for every comparison; 42 of 143 (29.4%) fall within the 3-point tolerance under one uniform protocol.

**Scope and limits**

- Not peer reviewed. The counts above are counts of datasets, not significance tests, and the near-tie with a plain random forest does not show equivalence. The 20 datasets were also used while the method was developed, so they are not an independent test; they are 18 distinct feature collections (CARGOOD and CARVGOOD share one feature table and differ only in which class is positive, and so do YEAST5 and YEAST5ERL).
- Model capacity is unequal (ERF 1,000 trees, the forests 100 trees, CSRBoost 50 stumps). A tree-count-matched comparison exists; no matched-compute study has been completed. ERF's median fit-and-score time is 6.6 to 35.6 times that of a plain random forest on the five new datasets.
- The gain over CSRBoost is concentrated in four datasets (CARGOOD, satimage, thyroid_sick, mammography: 65.7% of the summed gain), a plain random forest gains +0.0693 over CSRBoost, and the cause is untested.
- Software environments differ (the June runs used an unrecorded environment; the 2,000 new fits ran on Python 3.12.14, 800 fits, and 3.13.18, 1,200 fits). Prediction-level checks, per-fit records and the code that ran the 2,000 fits are private, so those fits cannot be rerun from the public repository.

Detail, the three further experiments, related work and corrections: [research/RESEARCH-EVIDENCE.md](research/RESEARCH-EVIDENCE.md) and the [November record](research/results/november2026_aligned_extension/README.md). The application uses its own evaluation procedure and does not reproduce the research tables.

- [FixYourDataset application](https://fixyourdataset.fly.dev)
- [Separate CSRBoost reproducibility audit](https://github.com/SherwanAli0/csrboost-audit)
- MIT for code and CC BY 4.0 for original documentation/results, with third-party exceptions described below.

Contents: [1 ERF-CSRBoost](#1-what-erf-csrboost-is) | [2 App and runner](#2-what-the-app-implements-and-how-it-differs-from-the-research-runner) | [3 Run the app](#3-run-the-app-locally) | [4 Run the research runner](#4-run-the-research-runner-and-obtain-the-data) | [5 Results](#5-results) | [6 Uploads and server state](#6-what-happens-to-uploaded-files-and-server-state) | [7 Files and licences](#7-files-and-licences) | [8 Limitations](#8-limitations) | [9 Audit](#9-relationship-to-the-audit) | [10 Citation](#10-citation) | [11 Provenance](#11-provenance-of-this-repository) | [Acknowledgements](#acknowledgements)

## 1. What ERF-CSRBoost is

ERF-CSRBoost builds on CSRBoost (Yadav et al., IEEE Access, 2025), a method for imbalanced
classification that balances the training data by clustered undersampling of the majority class
followed by SMOTE, and then fits AdaBoost.

ERF-CSRBoost keeps a CSRBoost-style balancing step but replaces the booster with a bagged ensemble
of random forests. For each of 20 bags it clusters the majority class with MiniBatchKMeans (K equal
to the number of minority rows), keeps 50 percent of each cluster, adds SMOTE samples for the
minority class, and fits a 50-tree random forest. The 20 bags' predicted probabilities are averaged.
The decision threshold is 0.5.

- It is a bagged ensemble of resampled random forests. It is **not** a boosting method.
- Its balancing step is CSRBoost-style, not identical to the published step: it uses MiniBatchKMeans
  with one initialisation and has no class-redefinition guard (`docs/IMPLEMENTATION_MAP.md`).
- Research status: experimental and not peer reviewed. A fixed number of trees and bags was used with no
  tuning; whether it scores higher depends on the dataset (section 5).

## 2. What the app implements and how it differs from the research runner

FixYourDataset is a single-page web app with a small Python server (standard library
`http.server`, scikit-learn, imbalanced-learn). You load a bundled sample or upload a file (30 MB
limit). The file picker lists `.csv`, `.xlsx`, `.xls`, `.data`, `.dat`, `.names`, `.txt`, `.tsv` and
`.tab`. The server's conversion step also reads JSON, Stata (`.dta`) and ZIP or GZIP archives of
these (read from the code); the picker does not list them, so drop such a file onto the upload
area or send it to the API (section 3). The app detects the delimiter and the target column and
then:

1. Splits first, resamples the **training** split only, and never touches the held-out test rows.
2. Returns a balanced training file, the untouched test file, an illustrative pipeline snippet and a
   receipt of what was done.
3. Compares three models on the same held-out rows: a plain random forest with class weights,
   CSRBoost (a port of the published recipe), and ERF-CSRBoost. Confusion matrices, a threshold
   slider (exploratory, see below) and a headline verdict that includes the cases where a method
   does not help.
4. Runs a "Leakage Lab" that shows what happens when you resample before splitting, a ranked risk
   table on unseen rows, and a "score your own file" option.

Two points about leakage handling: the leaky arm of the Leakage Lab resamples all rows on purpose,
and "score your own file" trains on every labelled row including the held-out ones, by design. Neither
feeds the reported test metrics.

**The app's ERF is adapted from the research runner. It is not numerically identical to it, and the
app does not reproduce the research tables.** It keeps the bag count (20), tree count (50), the seed
rule and the probability averaging. It differs in preprocessing (imputation, one-hot columns and
SMOTE-NC for mixed data), in the scaling used for SMOTE, in the fallback when SMOTE fails, in the
training-size cap (4,000 rows in the single-split comparison) and in evaluation (one stratified
75/25 split by default, against 100 repeated cross-validation folds in the research runner). The
full map is in [`docs/IMPLEMENTATION_MAP.md`](docs/IMPLEMENTATION_MAP.md).

The comparison panel is illustrative: one split, no uncertainty estimate, models of unequal
capacity (a 200-tree forest, 1,000 ERF trees, an AdaBoost of 50 trees). The threshold slider is
exploratory because it uses the labelled test rows; the headline metrics and the verdict stay at 0.50.

## 3. Run the app locally

Needs Python 3.11. The commands below were run on Windows 11 with Python 3.11.9; Linux and macOS
were not tested.

Windows (PowerShell):

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python backend\server.py
```

macOS or Linux:

```
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python backend/server.py
```

Then open http://localhost:8600 (set the `PORT` environment variable to use another port).

**Network exposure.** The server listens on all network interfaces (`0.0.0.0`), so while it runs it
is reachable from other devices on the same network at your machine's address, not only at
`localhost`. It has no login, and there is no setting for the listening address; this release does
not change the code. To keep it off the network, run it only on a network you trust, block the port
in your firewall, or run the Docker image with the port published to the loopback address only
(for example `docker run -p 127.0.0.1:8600:7860 <image>`; the image listens on port 7860). The
Docker route was not tested here.

`requirements.txt` pins pandas 2.3.3, numpy 2.2.6, scikit-learn 1.8.0 and imbalanced-learn 0.14.1,
and lists openpyxl and xlrd for Excel files. numpy is pinned to the tested version.

Run the tests from the repository root, with the virtual environment's interpreter (Windows shown;
on macOS or Linux use `.venv/bin/python`). If you activate the environment first, `python` is that
interpreter and the same commands work with `python` in place of the path. The two `node` tests need
Node 18 or newer and no packages.

```
.venv\Scripts\python backend/test_head.py
.venv\Scripts\python backend/test_rare_split_guard.py
.venv\Scripts\python backend/test_small_fixes.py
.venv\Scripts\python backend/test_sample_cards.py
.venv\Scripts\python backend/test_torture.py        # several minutes
node frontend/test_verdict.js
node frontend/test_receipt.js
```

What these tests check, and what was not checked, is in
[`docs/LIMITATIONS_AND_VERIFICATION.md`](docs/LIMITATIONS_AND_VERIFICATION.md).

The app runs on Fly.io as `fixyourdataset` (Frankfurt, one always-on machine). `DEPLOY_FLY.md`
describes the update and rollback procedure. The Dockerfile binds the app to `$PORT`.

### API

The page talks to the server through a small JSON interface that you can also call directly. It is
the app's own interface, not a versioned or stable API, and it has no authentication. Request
bodies are JSON; a failed POST returns status 400 with `{"ok": false, "error": "..."}`. Responses
carry `Cache-Control: no-store`. The source of truth is the docstring and handlers in
[`backend/server.py`](backend/server.py).

| Method and path | Purpose |
|---|---|
| `GET /` | The single-page app (`frontend/index.html`) |
| `GET /healthz` (also `/health`) | Returns `ok`; the Fly.io health check |
| `GET /api/samples` | List of the four bundled sample datasets |
| `GET /api/sample?name=<file>` | CSV text of one bundled sample |
| `POST /api/convert` | `{b64, name}`: decodes an uploaded file (text encodings, ZIP or GZIP, Excel, JSON, Stata) and returns `{ok, csv, notes}`; 30 MB limit |
| `POST /api/profile` | `{csv}`: column profile and suggested target column |
| `POST /api/analyze` | `{csv, target, method, do_classify, classifier, folds, src}`: split first, resample the training split only, compare the three models, return the balanced and test files and the receipt; `folds` is clamped to 1 to 100 |
| `POST /api/score` | `{train_csv, target, method, classifier, score_csv}`: train on every labelled row and return a risk score and flag for each new row |
| `GET /stats`, `GET /whoami/<key>` | Owner-only usage page and owner cookie; return 404 unless the owner's `STATS_KEY` is given (section 6) |

## 4. Run the research runner and obtain the data

The research runner is `research/scripts/comparison_rf_csrboost_erf.py`. It compares CSRBoost
(re-implemented), ERF-CSRBoost, RF-Raw (a plain random forest) and RF-BRF (a balanced random
forest) on 15 datasets with 5-fold stratified cross-validation repeated 20 times (100 folds per
dataset), identical folds for every method, a scaler fitted on the training fold only, and a fixed
0.5 threshold. The ERF part is described in section 1.

**Data are not included.** Download each file from its original provider (table below) and place
it exactly as shown under `research/`. The file hashes in `docs/RESULTS_EVIDENCE.md` let you check
that a download is the same file the stored results used.

| Folder and file to create | Dataset (run key) | Where to get it | Terms |
|---|---|---|---|
| `research/01_PSDAS/data.csv` | Predict Students' Dropout and Academic Success (PSDAS) | UCI, https://doi.org/10.24432/C5MC89 | CC BY 4.0 on the UCI page |
| `research/02_ESR/Epileptic Seizure Recognition.csv` | Epileptic Seizure Recognition (ESR) | UCI (derived from University of Bonn recordings) | Not established; read the original source's terms |
| `research/03_DCCC/DCCC.xls` | Default of Credit Card Clients (DCCC) | UCI, https://doi.org/10.24432/C55S3H | CC BY 4.0 on the UCI page |
| `research/04_BCW/wdbc.data` | Breast Cancer Wisconsin, diagnostic (BCW) | UCI | CC BY 4.0 on the UCI page |
| `research/05_ESDRP/diabetes_data_upload.csv` | Early Stage Diabetes Risk Prediction (ESDRP) | UCI | CC BY 4.0 on the UCI page |
| `research/06_CB/sonar.all-data` | Connectionist Bench, Sonar (CB) | UCI | CC BY 4.0 on the UCI page |
| `research/07_GLASS/glass.csv` | Glass Identification (GLASS, the 196-row file) | UCI | CC BY 4.0 on the UCI page |
| `research/08_ILPD/ilpd.csv` | Indian Liver Patient Dataset (ILPD) | UCI, https://doi.org/10.24432/C5D02C | CC BY 4.0 on the UCI page |
| `research/09_SEED/seeds_dataset.txt` | Seeds (SEED) | UCI | CC BY 4.0 on the UCI page |
| `research/10_WINE/wine.data` | Wine (WINE) | UCI | CC BY 4.0 on the UCI page |
| `research/11_YEAST5/yeast5.dat` | Yeast, KEEL imbalanced file yeast5 (YEAST5) | KEEL imbalanced datasets, https://sci2s.ugr.es/keel/imbalanced.php | No terms stated by KEEL |
| `research/12_YEAST5-ERL/yeast5erl.dat` | Yeast, ERL class against the rest, KEEL-format file named yeast5-ERL (YEAST5ERL) | KEEL | No terms stated by KEEL |
| `research/13_CARGOOD/car-good.dat` | Car Evaluation, "good" (CARGOOD) | KEEL | No terms stated by KEEL |
| `research/14_CARVGOOD/car-vgood.dat` | Car Evaluation, "very good" (CARVGOOD) | KEEL | No terms stated by KEEL |
| `research/15_FLARE-F/flare-F.dat` | Solar Flare, class F (FLAREF) | KEEL | No terms stated by KEEL |

UCI datasets are on https://archive.ics.uci.edu/ (search by name). The KEEL files are processed
versions of UCI data, and the UCI licence is not assumed to cover them. Details and caveats are in
`THIRD_PARTY_NOTICES.md`. Which exact file versions and label encodings the stored run used is
recorded in `docs/RESULTS_EVIDENCE.md`.

**Run.** After installing `requirements.txt` (section 3; SciPy comes with scikit-learn, and xlrd is
needed for the DCCC `.xls`):

```
python research/scripts/comparison_rf_csrboost_erf.py --datasets BCW CB --splits 5 --repeats 20
```

Valid dataset keys are `PSDAS ESR DCCC BCW CB ESDRP ILPD GLASS SEED WINE YEAST5ERL CARGOOD CARVGOOD
YEAST5 FLAREF`. With no `--datasets` argument all 15 run. Output goes to
`research/outputs/comparison_<timestamp>/` as `fold_results.csv` and `dataset_summary.csv`; the
script prints its tables to the console and writes no log file (redirect the output if you want one).

Caveats, from reading the script (the commands above were **not executed** during release
preparation, and no training was run):

- The script finds its data and writes its output relative to the folder one level above
  `scripts/`. Keep it at `research/scripts/`. If you copy it elsewhere, data are looked up and
  `outputs/` is created relative to that new parent folder.
- The full 15-dataset run took about 10 hours on the author's machine in April 2026 (single
  recorded run). ESR and DCCC are the largest.
- The run keeps nothing if it is interrupted; results are written when the run finishes.
- The versions of Python and the libraries used for the stored April and June runs were not recorded, so exact
  agreement with the stored numbers is not promised. (The November extension recorded its library versions in
  `research/results/november2026_aligned_extension/protocol.json`; its Python versions are in private per-fit files.)
  Small datasets reproduced exactly between two of the author's runs; the two largest (ESR and DCCC) differed
  slightly, for a reason that was not established.
- The April run was produced by an earlier version of this script, whose text was not preserved. The
  loaders were corrected afterwards (ESR and DCCC paths, and the Sonar class polarity), and the ESR,
  DCCC and CB rows of the June record come from reruns with the corrected script included here.
  Running the selected script on CB gives the corrected CB values (positive class R), which match the
  June record, not the April run (section 5).
- The runner does not stop on some failures: a dataset that fails to load is logged and skipped, a
  method that raises in a fold is stored as NaN and left out of the summary means, and a failed SMOTE
  call silently falls back to the rows without oversampling (`research/README.md`, section 6). A run
  that finishes is therefore not proof of a complete experiment: check that `fold_results.csv` has 100
  rows per dataset and method and no NaN. The stored June record has all 6,000 expected rows and no
  NaN; whether a SMOTE fallback occurred in it was not established.
- The exploratory 18-dataset run in section 5 was made with a script that is not included. It cannot
  be reproduced from this repository.
- The November extension (2,000 fits) was produced by orchestration code (`run.py`, `worker.py`, `legacy_adapter.py`)
  that is not in this repository, so those fits cannot be rerun from here. Its `legacy_source.py` is the runner in
  `research/scripts/` (same SHA-256, 3a799cfa...). The checker in `research/results/november2026_aligned_extension/`
  verifies the saved tables and checksums only and runs no model.
- To compare with stored numbers, use the files in `research/results/` (section 5) and the check in
  `docs/RESULTS_EVIDENCE.md`, section 10.

## 5. Results

**Which numbers are which.** The tables below are the **June four-way record** (historical, AUC-based; the latest AP-based comparison is the November record above): 15 datasets x
4 methods x 100 folds = 6,000 records, the record behind the author's graduation thesis tables. It
was **recomputed** here from its saved per-fold records. It is the 28 April 2026 run
(`comparison_20260428_141315`, kept in `research/results/` as the earlier run) with the rows of three
datasets, ESR, DCCC and CB, replaced by June 2026 reruns; for CB the positive class was corrected from
the larger class (M) to the smaller class (R). Those historical records were not retrained for the original release; the November extension above contains 2,000 newly fitted records. The records
are in `research/results/comparison_june2026_four_way/`. A separate exploratory run on 18 datasets is
described at the end of this section; it is an exploratory supplement.

**Read these limits first.** This is an exploratory comparison on 15 datasets that were also used while developing the method (the
capacity and development-dataset limits of Scope and limits at the top of this file apply), with a fixed 0.5 threshold and untuned
baselines. No equal-capacity comparison was run on these 15 datasets. The "CSRBoost" here is a local re-implementation, not the
authors' code. Several datasets contain duplicate rows (notably ESDRP and FLARE-F), CARGOOD and CARVGOOD share features, YEAST5 and
YEAST5ERL share their whole feature set (different target labels), and YEAST5ERL has only 5 positives, so the 15 dataset labels are
not 15 separate feature collections and the effective number of independent datasets is below 15. Many test rows also have an exact
feature twin in the training rows: in the first fold of the runner's splitter, 79 of 104 ESDRP test rows and 177 of 214 FLARE-F test
rows (a one-split diagnostic, `docs/RESULTS_EVIDENCE.md`, section 3.5). This is not an independent evaluation and does
not establish a general improvement.

Mean AUC by dataset (mean over 100 folds):

| Dataset | CSRBoost | ERF-CSRBoost | RF-Raw | RF-BRF |
|---|---|---|---|---|
| PSDAS | 0.7601 | 0.8212 | 0.8081 | 0.8134 |
| ESR | 0.9777 | 0.9960 | 0.9952 | 0.9946 |
| DCCC | 0.7630 | 0.7710 | 0.7637 | 0.7727 |
| BCW | 0.9922 | 0.9911 | 0.9897 | 0.9906 |
| CB | 0.9049 | 0.9319 | 0.9276 | 0.9278 |
| ESDRP (duplicate rows) | 0.9753 | 0.9984 | 0.9985 | 0.9983 |
| ILPD | 0.7276 | 0.7537 | 0.7433 | 0.7511 |
| GLASS (196-row file) | 0.9514 | 0.9759 | 0.9755 | 0.9741 |
| SEED | 0.9833 | 0.9802 | 0.9795 | 0.9775 |
| WINE | 0.9993 | 1.0000 | 1.0000 | 0.9998 |
| YEAST5ERL (5 positives) | 0.9996 | 1.0000 | 1.0000 | 0.9992 |
| CARGOOD | 0.9760 | 0.9933 | 0.9954 | 0.9958 |
| CARVGOOD | 0.9972 | 0.9999 | 0.9999 | 0.9998 |
| YEAST5 | 0.9853 | 0.9912 | 0.9879 | 0.9913 |
| FLAREF (duplicate rows) | 0.9073 | 0.8705 | 0.8378 | 0.8972 |

Global means (mean of the 15 dataset means):

| Method | ACC | AUC | F1 | AP | G-Mean |
|---|---|---|---|---|---|
| CSRBoost (live) | 0.8955 | 0.9267 | 0.7248 | 0.7618 | 0.8707 |
| ERF-CSRBoost | 0.9188 | 0.9383 | 0.7558 | 0.8151 | 0.8441 |
| RF-Raw | 0.9261 | 0.9335 | 0.6749 | 0.8158 | 0.7256 |
| RF-BRF | 0.9016 | 0.9389 | 0.6952 | 0.8103 | 0.8918 |

What these records show, with the dataset as the unit of analysis (N = 15):

- **Higher AUC than CSRBoost.** ERF-CSRBoost has a higher mean AUC than the re-implemented CSRBoost
  on 12 of 15 datasets and a lower one on the other 3 (BCW, SEED and FLAREF; no ties), with a
  global mean AUC of 0.9383 against 0.9267 (mean difference +0.0116). The Wilcoxon signed-rank test
  across datasets gives p = 0.030, not adjusted for multiple comparisons; the Friedman test over the
  four methods gives p = 0.005. Read these p-values with the limits stated above (unequal capacity and development datasets in Scope and limits;
  effective N below 15 in 'Read these limits first') and one more: the Holm family of three comparisons (ERF-CSRBoost
  against each other method) was chosen after seeing the data. It gives 0.060 for the CSRBoost comparison; with all six pairwise comparisons the Holm-adjusted p is 0.121. The result is
  borderline: it is p = 0.057 if CB and YEAST5ERL are removed. Clear per-dataset gains over CSRBoost,
  with agreement across folds, are on PSDAS, ESR, DCCC and CARGOOD (and on ESDRP, which is
  duplicate-driven).
- **Ties.** ERF-CSRBoost is not distinguishable from the balanced random forest RF-BRF on AUC (mean
  difference -0.0006, p = 0.15). On datasets where every method scores about 0.97 or higher
  ERF-CSRBoost cannot be separated from the random forests, and the 12 of 15 count above includes
  near-ties there (WINE, YEAST5ERL and CARVGOOD, higher by less than 0.005).
- **Loses.** On FLAREF, ERF-CSRBoost is below CSRBoost by 0.037 AUC (and below RF-BRF by 0.027). On
  BCW and SEED CSRBoost is slightly higher, by margins that are not meaningful. The mean G-Mean at
  the 0.5 threshold is lower than for CSRBoost and for RF-BRF (not significant across datasets). YEAST5ERL
  and FLAREF drive the CSRBoost gap (without them ERF-CSRBoost is higher by +0.0117), but the gap to
  RF-BRF stays at -0.0038 without them.
- **Not shown.** Why ERF-CSRBoost scores higher where it does is untested: no ablation separates the change
  of classifier, the bagging and the larger number of trees.

Earlier run and CB note: the April run is the same record except for the ESR, DCCC and CB rows. In
the April run CB used the larger class as the positive class and gave CSRBoost 0.9028 and
ERF-CSRBoost 0.9362 for CB, against 0.9049 and 0.9319 in the June record. ESR and DCCC were rerun
and their mean AUC differs from April by at most 0.0001 for every method; why a rerun of those two
large datasets does not reproduce the 28 April rows exactly was not established.

### Extended exploratory run (18 datasets)

**Exploratory supplement.** A later run covers 18 datasets, 100 folds each: 13 of
the 15 datasets above (ESR and DCCC were skipped) plus five external datasets (mammography,
thyroid_sick, coil_2000, satimage and a credit-card fraud subsample), with CSRBoost, ERF-CSRBoost and
a balanced random forest of 1,000 trees (BRF), whose tree count matches ERF-CSRBoost's. Its protocol
differs from the June record: every method is trained on 70 percent of each outer training fold,
not on the whole fold. The run's script is **not included**, so it cannot be reproduced from this
repository; the published file is its raw per-fold records, with the rows for further models, which
are not part of this release, removed (rule, hashes and row counts in
`docs/RESULTS_EVIDENCE.md`, section 8.1). The five added datasets were chosen by the author and no
selection rule is recorded. Dataset-level results (N = 18, exact two-sided Wilcoxon, **uncorrected**,
six tests shown):

| Comparison (unit = dataset, N = 18) | Mean difference | Datasets higher / lower | Bootstrap 95% interval of the mean | Wilcoxon exact, two-sided p |
|---|---|---|---|---|
| AUC: ERF-CSRBoost minus CSRBoost | +0.0146 | 15 / 3 | [+0.0062, +0.0230] | 0.0023 |
| AUC: ERF-CSRBoost minus BRF-1000 | -0.0030 | 10 / 8 | [-0.0084, +0.0010] | 0.9323 |
| AP: ERF-CSRBoost minus CSRBoost | +0.0694 | 16 / 2 | [+0.0361, +0.1062] | 0.0003 |
| AP: ERF-CSRBoost minus BRF-1000 | +0.0010 | 12 / 6 | [-0.0171, +0.0180] | 0.3038 |
| G-Mean at 0.5: ERF-CSRBoost minus CSRBoost | -0.0323 | 8 / 10 | [-0.0777, +0.0048] | 0.6705 |
| G-Mean at 0.5: ERF-CSRBoost minus BRF-1000 | -0.0645 | 8 / 10 | [-0.1323, -0.0126] | 0.0304 |

- Against CSRBoost, ERF-CSRBoost has a higher mean average precision on 16 of 18 datasets. The capacity and development-dataset
  limits of Scope and limits apply (13 of the 18 datasets are development datasets), so this does not show that ERF-CSRBoost is a
  better method.
- Against the tree-count-matched BRF there is no AUC or AP advantage, and ERF-CSRBoost has a lower
  G-Mean at the 0.5 threshold. No other equal-capacity result exists.
- The gains concentrate in a few datasets (CARGOOD, satimage, thyroid_sick; YEAST5ERL, with 5
  positives, is fourth by AP gain). Without the first three the AP difference over CSRBoost is +0.0421
  (13 of 15 higher) and over BRF +0.0055 (10 of 15 higher, p 0.25); without all four it is +0.0363
  over CSRBoost and -0.0011 over BRF.
- The numbers of this run and of the June record use different protocols and are never
  combined here.

Per-dataset table, protocol differences, the filter that produced the published file and every
caveat are in [`docs/RESULTS_EVIDENCE.md`](docs/RESULTS_EVIDENCE.md), section 8. The full evidence,
protocol, per-dataset tests, the status of each claim and the limits for the June record are in
the same document.

## 6. What happens to uploaded files and server state

- **Upload contents** are decoded and processed in memory. The server writes no temporary file and no
  cache of an upload, and the outputs (balanced file, test file, scored file) are returned in the
  response and not kept. Responses carry `Cache-Control: no-store`. Your browser keeps the data in a
  page variable while the page is open. (Checked by reading the code and by running the server through
  an upload, profile, analysis and scoring request and listing the files it created.)
- **A usage log is stored.** The server keeps a SQLite file with one row per event: timestamp, UTC
  day, kind (`visit`, `analyze` or `score`), a short detail (the demo file name or the word `upload`,
  plus the fold count), the HTTP referrer exactly as the browser sent it (up to 200 characters, so it
  can include a query string), an owner flag, and a visitor tag. The tag is the first 12 hex
  characters of a SHA-256 hash of the IP address, the user agent, the day and a server secret
  (`STATS_KEY`). It changes daily and is pseudonymous, not anonymous. Not stored: the IP address
  itself, names, file contents, file names, column names. Visitors receive no cookie.
- **Weaker if `STATS_KEY` is unset.** Without the secret the hash key is a constant, and an IP
  address could be guessed back from a tag if the database ever leaked. Set `STATS_KEY` on any
  deployment (`DEPLOY_FLY.md`). Whether it is set on the live app was not verified.
- **Retention.** There is no purge job and no deletion route. Events stay until the database file or
  the volume holding it is deleted. On Fly the file lives on the `/data` volume; locally it is
  `data_local/usage.db` (ignored by Git).
- **Platform logs.** The hosting provider's own access logs are outside the application and were
  not inspected.
- An owner-only statistics page exists and returns 404 without the owner's key.

## 7. Files and licences

| Path | What it is | Licence |
|---|---|---|
| `backend/` | Python server, engine, models (plain baseline, CSRBoost port, ERF-CSRBoost), usage log, tests. `gen_samples.py` writes three unbundled synthetic example files; it does not regenerate `samples/` | MIT |
| `frontend/` | `index.html` (single-page app) and its two Node test files | MIT |
| `research/scripts/` | The research runner | MIT |
| `research/results/` | Per-fold records (and, for the June and April records, per-dataset summaries) of the June four-way record, the earlier April run and an exploratory 18-dataset run (metrics only; the exploratory file is a filtered copy and its script is not included, see `research/README.md`) | CC BY 4.0 |
| `samples/*.csv` | Four modified copies of UCI datasets | Third party: UCI, CC BY 4.0, changes listed |
| `samples/README.md`, `docs/`, `README.md`, `research/README.md`, `DEPLOY_FLY.md`, `THIRD_PARTY_NOTICES.md` | Documentation | CC BY 4.0 |
| `Dockerfile`, `fly.toml`, `.dockerignore`, `requirements.txt` | Deployment and dependency files | MIT |
| `.gitignore`, `.gitattributes`, `research/.gitignore`, `research/MANIFEST.sha256` | Configuration files and the SHA-256 list for the stored research files | MIT |
| `LICENSE`, `LICENSE-CC-BY-4.0.txt`, `CITATION.cff` | Licence texts and citation metadata | n/a |
| Dataset files you download for `research/` | Not included | Third party, own terms |

`LICENSE` states the scope precisely; `LICENSE-CC-BY-4.0.txt` is the unmodified official legal
code. No licence here adds a mandatory citation condition. Third-party material, with credit and
the changes made to the samples, is listed in `THIRD_PARTY_NOTICES.md`.

**Licence scope.** These licences cover this repository only. The author's graduation-project working folder is not published here; it carries its own `LICENSE` (CC BY-NC-ND 4.0) and is not covered by the MIT or CC BY 4.0 terms above.

## 8. Limitations

The main ones, with the full list in `docs/LIMITATIONS_AND_VERIFICATION.md`:

- The limits of the November comparison (unequal capacity, development datasets, concentrated gain, environments, private artefacts)
  are in Scope and limits at the top of this file and in `research/RESEARCH-EVIDENCE.md`; those of the June record are in section 5
  ("Read these limits first"). Beyond those: a fixed threshold, untuned baselines, library versions of the April and June runs not
  recorded, and an 18-dataset run that is uncorrected, uses a different protocol and cannot be reproduced from this repository (no
  script).
- Data quality of the research datasets (duplicate rows, test rows with an exact feature twin in the training rows, datasets that
  share a whole feature set) is described in section 5; the saved results were not deduplicated. The GLASS file has 196 rows, not the
  214 of the original UCI data, so a rerun on the 214-row file would not be the same experiment
  (`docs/LIMITATIONS_AND_VERIFICATION.md`).
- A finished run of the research runner is not proof of a complete experiment (failures are skipped or
  stored as NaN, and a failed SMOTE call falls back silently); whether a fallback occurred in the stored
  June record was not established.
- The app is not numerically equal to the research ERF and uses a single split by default.
- The tests cover synthetic and bundled data and a set of messy-input cases; browser rendering of
  some screens, the live deployment beyond a health request, and the documented research command
  were not verified.
- In a small two-seed pilot, removing clustering did not lower mean AP while a one-bag variant was lower on 4 of 5 dataset families; neither has been tested at scale.
- Not a production service.

## 9. Relationship to the audit

The [audit](https://github.com/SherwanAli0/csrboost-audit) is a separate project that reproduces and
investigates the published CSRBoost method. This repository is a different project: an experimental
extension of that method and a web application. They have separate histories and scopes and no
blended result claims. The audit's figures (for example 42 of 143 metric-algorithm combinations
under a uniform protocol) are about the audit only and say nothing about ERF-CSRBoost; the numbers
in this repository say nothing about the audit's questions.

## 10. Citation

Cite this software using [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository"
button). Also cite the CSRBoost paper that ERF-CSRBoost extends:

```bibtex
@article{yadav2025csrboost,
  author  = {Yadav, Seema and Joshi, Dhruvanshu and Mulye, Soham and Asari, Labib and Udmale, Sandeep S. and Bhole, Girish P.},
  title   = {{CSRBoost}: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification},
  journal = {IEEE Access},
  volume  = {13},
  pages   = {175078-175093},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3616207}
}
```

Please also cite CSRBoost (S. Yadav, D. Joshi, S. Mulye, L. Asari, S. S. Udmale and G. P. Bhole, "CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification," IEEE Access, vol. 13, pp. 175078-175093, 2025, doi 10.1109/ACCESS.2025.3616207); this is scholarly credit, not a licence condition. Contacting the paper's authors is optional. There is no DOI, venue or peer-reviewed publication for ERF-CSRBoost.

## 11. Provenance of this repository

This repository starts with a single curated commit and no earlier history. The application was
developed in a private repository, and that first commit is a copy of the private repository's
release state; its earlier commits are not included and their dates are not reproduced here. The
application and the research it builds on are older than this repository. The `research/` folder is
copied unchanged from the author's research working folder, except for one file that is derived from
a larger raw file by a stated filter: see [`research/README.md`](research/README.md)
for what it contains, how to run the runner, and `research/MANIFEST.sha256` for the SHA-256 of each file.

## Acknowledgements

The ERF-CSRBoost research was carried out as a graduation project at Uskudar University under the
supervision of Dr. Gamze Uslu. This is an acknowledgement, not a licence condition and not a claim
of institutional endorsement. The sample and research datasets are the work of their original
creators, credited in `THIRD_PARTY_NOTICES.md`.
