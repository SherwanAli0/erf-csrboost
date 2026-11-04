# ERF-CSRBoost and FixYourDataset

An experimental ensemble extension of CSRBoost, with a web application for exploring class imbalance and comparing model behavior. Developed alongside a separate reproducibility audit of the published CSRBoost method.

**Status.** ERF-CSRBoost is an unpublished research prototype. It has not been peer reviewed, and
nothing in this repository claims that it is a proven or general improvement over any other method.
The numbers below are an exploratory comparison with stated limits.

- Live application: https://fixyourdataset.fly.dev
- Separate audit of the published CSRBoost method: https://github.com/SherwanAli0/csrboost-audit
- Licences: MIT for code, CC BY 4.0 for original documentation and result files, with third-party
  exceptions (section 7).

Contents: [1 ERF-CSRBoost](#1-what-erf-csrboost-is) | [2 App and runner](#2-what-the-app-implements-and-how-it-differs-from-the-research-runner) | [3 Run the app](#3-run-the-app-locally) | [4 Run the research runner](#4-run-the-research-runner-and-obtain-the-data) | [5 Results](#5-results) | [6 Uploads and server state](#6-what-happens-to-uploaded-files-and-server-state) | [7 Files and licences](#7-files-and-licences) | [8 Limitations](#8-limitations) | [9 Audit](#9-relationship-to-the-audit) | [10 Citation](#10-citation)

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
- Research status: experimental and unpublished. A fixed number of trees and bags was used with no
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
variants of UCI data, and the UCI licence is not assumed to cover them. Details and caveats are in
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
- The versions of Python and the libraries used for the stored run were not recorded, so exact
  agreement with the stored numbers is not promised. Small datasets reproduced exactly between two
  of the author's runs; the two largest (ESR and DCCC) differed slightly, for a reason that was not
  established.
- The stored results come from an earlier version of this script. The loaders were corrected
  afterwards (ESR and DCCC paths, and the Sonar class polarity). Running the selected script on CB
  gives the corrected CB values, not the April ones (section 5).
- To compare with stored numbers, use the files in `research/results/` (section 5) and the check in
  `docs/RESULTS_EVIDENCE.md`, section 10.

## 5. Results

**Which numbers are which.** The tables below are a **historical artifact**: a run from 28 April
2026 (run `comparison_20260428_141315`, 15 datasets x 4 methods x 100 folds = 6,000 records) that was
**recomputed** here from its saved per-fold records. Nothing was retrained for this repository. The
records are in `research/results/comparison_20260428_141315/`.

**Read these limits first.** It is an exploratory comparison on 15 datasets that were also used while
developing the method, with a fixed 0.5 threshold and untuned baselines. ERF-CSRBoost averages 20
bags of 50-tree forests (1,000 trees), against 50 decision stumps for CSRBoost and 100 trees for
the random forest baselines; no equal-capacity comparison was run. The "CSRBoost" here is a local
re-implementation, not the authors' code. Several datasets contain duplicate rows (notably ESDRP and
FLARE-F), CB uses its larger class as the positive class in this run, and CARGOOD and CARVGOOD share
features, so the effective number of independent datasets is below 15. This is not an independent
evaluation and does not establish a general improvement.

Mean AUC by dataset (mean over 100 folds):

| Dataset | CSRBoost | ERF-CSRBoost | RF-Raw | RF-BRF |
|---|---|---|---|---|
| PSDAS | 0.7601 | 0.8212 | 0.8081 | 0.8134 |
| ESR | 0.9776 | 0.9960 | 0.9952 | 0.9946 |
| DCCC | 0.7631 | 0.7710 | 0.7637 | 0.7727 |
| BCW | 0.9922 | 0.9911 | 0.9897 | 0.9906 |
| CB (larger class positive) | 0.9028 | 0.9362 | 0.9276 | 0.9243 |
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
| CSRBoost (live) | 0.8951 | 0.9265 | 0.7262 | 0.7627 | 0.8701 |
| ERF-CSRBoost | 0.9185 | 0.9386 | 0.7579 | 0.8166 | 0.8432 |
| RF-Raw | 0.9259 | 0.9335 | 0.6776 | 0.8170 | 0.7252 |
| RF-BRF | 0.9009 | 0.9386 | 0.6971 | 0.8113 | 0.8910 |

What these records show, with the dataset as the unit of analysis (N = 15):

- **Higher AUC than CSRBoost.** ERF-CSRBoost has a higher mean AUC than the re-implemented CSRBoost
  on 12 of 15 datasets and a lower one on the other 3 (BCW, SEED and FLAREF; no ties), with a
  global mean AUC of 0.9386 against 0.9265. The Wilcoxon signed-rank test across datasets gives
  p = 0.030, not adjusted for multiple comparisons; the Friedman test over the four methods gives
  p = 0.005. Read these p-values with four limits. (1) The models have unequal capacity: 1,000
  trees for ERF-CSRBoost, 50 decision stumps for CSRBoost and 100 trees for the random forest
  baselines, and no equal-capacity comparison was run. (2) The 15 datasets were also used while the
  method was developed, so they are not a fresh test set. (3) The test treats the datasets as
  independent, but CARGOOD and CARVGOOD share features and some datasets contain duplicate rows, so
  the effective N is below 15. (4) The Holm family of three comparisons (ERF-CSRBoost against each
  other method) was chosen after seeing the data. It gives 0.060 for the CSRBoost comparison; with
  all six pairwise comparisons the Holm-adjusted p is about 0.12. The result is borderline: it is
  p = 0.057 if CB and YEAST5ERL are removed. Clear per-dataset gains over CSRBoost, with agreement
  across folds, are on PSDAS, ESR, DCCC and CARGOOD (and on ESDRP, which is duplicate-driven).
- **Ties.** ERF-CSRBoost is not distinguishable from the balanced random forest RF-BRF on AUC (mean
  difference -0.0001, p = 0.15). On datasets where every method scores about 0.97 or higher the
  methods cannot be separated.
- **Loses.** On FLAREF, ERF-CSRBoost is below CSRBoost by 0.037 AUC (and below RF-BRF by 0.027). On
  BCW and SEED CSRBoost is slightly higher, by margins that are not meaningful. The mean G-Mean at
  the 0.5 threshold is lower than for CSRBoost and for RF-BRF, driven by YEAST5ERL and FLAREF.
- **Not shown.** Why ERF-CSRBoost scores higher where it does is untested: no ablation separates the change
  of classifier, the bagging and the extra trees.

CB note: in this April run label 1 is the larger Sonar class. A later corrected rerun of CB and two
other datasets (ESR, DCCC; only CB changed materially) gives CSRBoost 0.9049 and ERF-CSRBoost 0.9319
for CB, against 0.9028 and 0.9362 in the table above.

The full evidence, protocol, per-dataset tests, the status of each claim and the limits are in
[`docs/RESULTS_EVIDENCE.md`](docs/RESULTS_EVIDENCE.md).

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
| `research/results/` | Per-fold and per-dataset scores of the April run, plus the per-fold records of the later June two-method rerun (metrics only) | CC BY 4.0 |
| `samples/*.csv` | Four modified copies of UCI datasets | Third party: UCI, CC BY 4.0, changes listed |
| `samples/README.md`, `docs/`, `README.md`, `research/README.md`, `DEPLOY_FLY.md`, `THIRD_PARTY_NOTICES.md` | Documentation | CC BY 4.0 |
| `Dockerfile`, `fly.toml`, `.dockerignore`, `requirements.txt` | Deployment and dependency files | MIT |
| `.gitignore`, `.gitattributes`, `research/.gitignore`, `research/MANIFEST.sha256` | Configuration files and the SHA-256 list for the stored research files | MIT |
| `LICENSE`, `LICENSE-CC-BY-4.0.txt`, `CITATION.cff` | Licence texts and citation metadata | n/a |
| Dataset files you download for `research/` | Not included | Third party, own terms |

`LICENSE` states the scope precisely; `LICENSE-CC-BY-4.0.txt` is the unmodified official legal
code. No licence here adds a mandatory citation condition. Third-party material, with credit and
the changes made to the samples, is listed in `THIRD_PARTY_NOTICES.md`.

## 8. Limitations

The main ones, with the full list in `docs/LIMITATIONS_AND_VERIFICATION.md`:

- Exploratory results on development datasets, unequal model capacity, fixed threshold, duplicate
  rows in some datasets, and library versions of the stored run not recorded.
- The app is not numerically equal to the research ERF and uses a single split by default.
- The tests cover synthetic and bundled data and a set of messy-input cases; browser rendering of
  some screens, the live deployment beyond a health request, and the documented research command
  were not verified.
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

The citation request is a request, not a licence condition. There is no DOI, venue or peer-reviewed
publication for ERF-CSRBoost.

## 11. Provenance of this repository

This repository starts with a single curated commit and no earlier history. The application was
developed in a private repository, and that first commit is a copy of the private repository's
release state; its earlier commits are not included and their dates are not reproduced here. The
application and the research it builds on are older than this repository. The `research/` folder is
copied unchanged from the author's research working folder: see [`research/README.md`](research/README.md)
for what it contains, how to run the runner, and `research/MANIFEST.sha256` for the SHA-256 of each file.

## Acknowledgements

The ERF-CSRBoost research was carried out as a graduation project at Uskudar University under the
supervision of Dr. Gamze Uslu. This is an acknowledgement, not a licence condition and not a claim
of institutional endorsement. The sample and research datasets are the work of their original
creators, credited in `THIRD_PARTY_NOTICES.md`.
