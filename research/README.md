# research/: runner script and stored results

ERF-CSRBoost is an experimental extension of CSRBoost and an unpublished research prototype. This
folder holds the script that produced the stored comparison and the saved per-fold records of three
result sets: the June four-way record (the headline record), the earlier April run, and an
exploratory 18-dataset run (raw fold records only, without its script). It holds no datasets, no
thesis or report documents, and no other research material.

**Nothing here was re-run for this repository.** No model was trained while preparing it. What was
checked is listed in section 2. The numbers are a historical artifact of an exploratory comparison
on 15 datasets that were also used while the method was developed, with a fixed 0.5 threshold,
untuned baselines and unequal model capacity (ERF-CSRBoost 1,000 trees, CSRBoost 50 stumps, the
random forest baselines 100 trees). Read [`../docs/RESULTS_EVIDENCE.md`](../docs/RESULTS_EVIDENCE.md)
before quoting any of them; it also contains the recomputed tables and the limits.

## 1. Contents

| Path | What it is | Licence |
|---|---|---|
| `scripts/comparison_rf_csrboost_erf.py` | The runner: CSRBoost (re-implemented), ERF-CSRBoost (plain version: 20 bags of 50-tree random forests), RF-Raw and RF-BRF, evaluated with 5-fold stratified cross-validation repeated 20 times (100 folds per dataset) | MIT |
| `results/comparison_june2026_four_way/fold_results.csv` | **The headline record (June four-way record):** 6,000 rows = 15 datasets x 4 methods x 100 folds, columns `dataset, method, fold, acc, auc, f1, ap, gmean`. It is the April run with the ESR, DCCC and CB rows replaced by June 2026 reruns | CC BY 4.0 |
| `results/comparison_june2026_four_way/dataset_summary.csv` | Per-dataset size, imbalance, and mean and standard deviation (population formula) of each metric for each method. Recomputed from the fold records and found equal | CC BY 4.0 |
| `results/comparison_june2026_four_way/global_means.csv` | Mean of the 15 dataset means per method and metric. Recomputed from the fold records and found equal | CC BY 4.0 |
| `results/comparison_20260428_141315/fold_results.csv` | **The earlier run (28 to 29 April 2026):** same layout, 6,000 rows. Its ESR, DCCC and CB rows were superseded by the June reruns (section 8). Kept for provenance | CC BY 4.0 |
| `results/comparison_20260428_141315/dataset_summary.csv` | Per-dataset summary of the April run | CC BY 4.0 |
| `results/exploratory_18_dataset_run/fold_results.csv` | **Exploratory supplement, not the headline:** 5,400 rows = 18 datasets x 3 methods (CSRBoost, ERF, BRF) x 100 folds, columns `dataset, src, fold, model, auc, ap, gmean_default, gmean_moved, pick`. A filtered copy of a 9,000-row file (section 8). Its script is not included | CC BY 4.0 |
| `MANIFEST.sha256` | SHA-256 of the seven files above. Check with `cd research` then `sha256sum -c MANIFEST.sha256` (or `Get-FileHash` in PowerShell). `.gitattributes` forces LF line endings on this list so the check also works after a Windows checkout with `core.autocrlf=true` | MIT (configuration file) |
| `.gitignore` | Keeps downloaded datasets and the runner's own outputs out of Git | MIT (configuration file) |

Six of the seven files were copied byte for byte from the author's research working folder (a
read-only source). The exploratory file is derived from its source by a fixed filter (section 8).
Each was compared by SHA-256 before and after copying, and again after the Git commit. They use CRLF line endings. The repository's `.gitattributes` marks them `-text` so no
checkout changes their bytes. If a tool does convert them, the LF-normalised hashes in
`../docs/RESULTS_EVIDENCE.md` section 11 apply instead.

The header comment of the runner is the author's original working text. It was left unchanged so
that the hash matches. It names the author's university and supervisor and describes ERF-CSRBoost
with the words "proposed novelty"; the current wording is in `../README.md`.

Some comments and docstrings in the runner are also out of date and were left as they are, because
the hash is pinned. The comment block at lines 155 to 173 says that stored CSRBoost values (the
`CSRBOOST_REPL` table, lines 175 to 191, which no code uses) are substituted in place of running
CSRBoost, the header (line 11), the docstring of `run_csrboost` (line 558) and a console line (line
691) say KMeans, and the console tables label the CSRBoost row "anchor" (lines 799 and 851). The code
does none of this: `run_csrboost` (line 556) fits CSRBoost live in every fold, with
`MiniBatchKMeans` (line 461, `n_init=1`, as the docstring at lines 458 to 459 says). Consistent with
that, the stored CSRBoost rows are not constants: in every dataset the 100 folds of the June record
hold at least 7 distinct AUC values (recomputed).

### How the runner relates to the stored records

The April run (28 to 29 April 2026) was produced by an earlier version of this script. The exact
April text was not preserved, so its identity with this file is inferred, not proven. This file is the
later, corrected version: on 20 June 2026 the author fixed three loader problems (the ESR path, the
DCCC path, and the class polarity of the Sonar dataset, CB), and the ESR, DCCC and CB rows of the
headline record come from June reruns with the corrected loaders (historical artifact, from the
author's records). The functions that implement ERF-CSRBoost are the same in this
file and in an older snapshot of the script (a comment in it refers to a reorganisation after 4 May
2026; compared by syntax tree in a review pass, not by this repository). Consequences:

- Running this script on CB gives the corrected CB values (positive class R), which match the June
  record, not the April ones (section 8).
- Running it on ESR and DCCC works; the older snapshot could not load those two datasets.
- The Python and library versions of the stored runs were not recorded.

The exploratory run in `results/exploratory_18_dataset_run/` was made with a different script that
imports this runner's CSRBoost and ERF-CSRBoost functions. That script is **not included**: the
exploratory run cannot be reproduced from this repository.

## 2. What was and was not checked

Checked during release preparation (read-only, no training):

- SHA-256 of every file against the values in this repository's documents and `MANIFEST.sha256`.
- The runner compiles (syntax check only, in a scratch copy, with Python 3.11).
- The per-fold records recompute to the numbers in `../docs/RESULTS_EVIDENCE.md`: for the June
  record, 6,000 rows with no missing values, the global means, ERF-CSRBoost higher than CSRBoost on 12
  of 15 datasets, the dataset-level Wilcoxon p of 0.030 (Holm 0.060), the Friedman p of 0.005, and the
  subset checks. The comparison with CSRBoost has unequal capacity (1,000 trees against 50 stumps) and
  uses datasets that were also used while the method was developed, so these are recomputed numbers,
  not evidence of a better method. The summary files of the June and April records recompute from their fold records.
  The June record equals the April run on 12 of 15 datasets row for row, and the exploratory file
  recomputes to the tables in section 8 of the evidence document.
- The feature twins, the shared feature sets and the GLASS row counts of section 4 were recomputed
  by loading the data files with the runner's loader logic, copied into a separate scratch script
  that is not part of this repository. The runner module was not imported or run and no model was
  fitted. The June record has all 6,000 expected rows (15 datasets x 4 methods x 100 folds) and no
  missing values (recomputed).
- The runner's paths and its error handling (section 6) were read from the code (sections 3 and 5).

**Not checked:** that the script runs at all in a fresh environment (it was never executed here),
that it runs with the library versions in `../requirements.txt`, that an unmodified download of each
dataset has the file layout the loaders expect, and that a fresh run reproduces the stored numbers.

## 3. Folder layout the runner expects

The runner computes `SCRIPT_DIR` as the folder one level above the folder that holds the script,
reads the data from numbered folders there, and writes its output there. With the layout of this
repository, that folder is `research/`:

```
research/
  scripts/comparison_rf_csrboost_erf.py     (the runner, keep it here)
  01_PSDAS/data.csv                         (you add the data folders, see section 4)
  02_ESR/Epileptic Seizure Recognition.csv
  ...
  outputs/comparison_<YYYYmmdd_HHMMSS>/     (created by the runner)
  results/                                  (stored records, never written by the runner)
```

If you copy the script somewhere else, for example into the repository root, the "one level above"
folder is the folder above the repository: the runner would look for data there and create an
`outputs/` folder there. Keep the script in `research/scripts/`.

## 4. Data (not included)

The data are not redistributed here. Download each file from its original provider and place it as
shown. UCI datasets are at https://archive.ics.uci.edu/ (search by name; DOIs for some are in
`../README.md` section 4). The KEEL files are from the KEEL imbalanced datasets page,
https://sci2s.ugr.es/keel/imbalanced.php. The terms of each source are summarised in
`../THIRD_PARTY_NOTICES.md`; the licence of the KEEL files and of the ESR file is not established, which is
why they are link-only. The five extra datasets of the exploratory run are not covered by the runner
and are not described as a download recipe here (`../THIRD_PARTY_NOTICES.md` names their sources).

| Run key | Place the file at | Source | Positive class used by the loader (read from the code) |
|---|---|---|---|
| PSDAS | `01_PSDAS/data.csv` (semicolon separated, column `Target`) | UCI, Predict Students' Dropout and Academic Success | `Enrolled` |
| ESR | `02_ESR/Epileptic Seizure Recognition.csv` (the loader takes the first `*.csv` in the folder) | UCI, Epileptic Seizure Recognition (derived from University of Bonn recordings). Terms not established: do not redistribute | label 1 against the rest |
| DCCC | `03_DCCC/DCCC.xls` (first `*.xls` or `*.xlsx`; header on the second row; first column dropped) | UCI, Default of Credit Card Clients | default = 1 (flipped if it were the majority) |
| BCW | `04_BCW/wdbc.data` | UCI, Breast Cancer Wisconsin (Diagnostic) | `M` |
| ESDRP | `05_ESDRP/diabetes_data_upload.csv` (column `class`) | UCI, Early Stage Diabetes Risk Prediction | `Negative` (the smaller class) |
| CB | `06_CB/sonar.all-data` | UCI, Connectionist Bench (Sonar) | `R` (rocks, 97 rows; June and later runs; the April run used `M`) |
| GLASS | `07_GLASS/glass.csv` (the 196-row file, type in the last column) | UCI, Glass Identification | type 7 |
| ILPD | `08_ILPD/ilpd.csv` (column `Class`) | UCI, Indian Liver Patient Dataset | `Class == 2` |
| SEED | `09_SEED/seeds_dataset.txt` | UCI, Seeds | class 1 |
| WINE | `10_WINE/wine.data` | UCI, Wine | class 3 |
| YEAST5 | `11_YEAST5/yeast5.dat` | KEEL imbalanced, yeast5 | `positive` |
| YEAST5ERL | `12_YEAST5-ERL/yeast5erl.dat` | KEEL imbalanced, file named yeast5-ERL (see `../README.md` section 4) | `positive` |
| CARGOOD | `13_CARGOOD/car-good.dat` | KEEL imbalanced, car-good | `positive` |
| CARVGOOD | `14_CARVGOOD/car-vgood.dat` | KEEL imbalanced, car-vgood | `positive` |
| FLAREF | `15_FLARE-F/flare-F.dat` | KEEL imbalanced, flare-F | `positive` |

Important caveats about the data:

- The loaders read the author's local copies by file name, column name and encoding. Whether an
  unmodified provider download has exactly this layout was **not** checked (for example, the ILPD
  loader needs a column named `Class`, and the GLASS loader expects a 196-row file, not the original
  214-row UCI file). Compare the file hashes in `../docs/RESULTS_EVIDENCE.md` section 3.1 if in doubt.
- If a data file is missing or fails to load, the runner prints `SKIP -- load error` for that
  dataset and carries on with the next one. After a run, check that `fold_results.csv` has 100 rows
  per dataset and method and no NaN values (section 6).
- CB, ESDRP, FLAREF and others have properties (class polarity, duplicate rows) that matter for the
  results; see `../docs/RESULTS_EVIDENCE.md` section 3.
- Some tasks share a whole feature collection. CARGOOD and CARVGOOD hold the same 1,728 feature rows
  (six features) and YEAST5 and YEAST5ERL hold the same 1,484 feature rows (eight features, in a
  different file order); in each pair only the target label differs (compared as sorted feature rows,
  recomputed). The 15 task labels are therefore not 15 separate feature collections, and the
  dataset-level tests still treat the tasks as independent.
- Exact feature twins across train and test. In the first fold of the runner's splitter
  (`RepeatedStratifiedKFold`, 5 splits, 20 repeats, `random_state` 42; lines 726 to 727 of the runner),
  the number of test rows whose full feature vector, whatever its class, also occurs in the training
  rows is 79 of 104 for ESDRP, 177 of 214 for FLAREF, 20 of 6,000 for DCCC, 4 of 117 for ILPD, 7 of
  297 for YEAST5 and 7 of 297 for YEAST5ERL (all recomputed). This is a one-split diagnostic, not an
  average over the 100 evaluations. FLAREF has 19 feature patterns that occur with both classes and
  DCCC has 21. Exact-feature overlap is different from resampling before splitting: the runner fits
  the scaler and the resampling on the training fold only, and the original data still contain the
  twins. The saved results were not deduplicated. Details: `../docs/RESULTS_EVIDENCE.md` section 3.5.
- GLASS: the selected file has 196 rows (type 3 absent, type 1 has 69 rows). The canonical UCI file has
  214 rows (type 1 has 70, type 3 has 17); it is not part of this repository. A rerun on the 214-row
  file would not be the same experiment as the stored GLASS rows.

## 5. Run

Python 3.11 was used by the author. The runner imports numpy, pandas, SciPy, scikit-learn and
imbalanced-learn, and needs xlrd (2.0.1 or newer) to read the DCCC `.xls`. From the repository root,
`pip install -r requirements.txt` installs these (SciPy comes with scikit-learn). Those pinned
versions are the application's; they have **not** been tested with the runner.

```
python research/scripts/comparison_rf_csrboost_erf.py --datasets BCW CB --splits 5 --repeats 20
```

On Windows with the launcher: `py -3.11 research\scripts\comparison_rf_csrboost_erf.py ...`.

Options (read from the code): `--datasets` takes run keys from the table above (default: all 15),
`--splits` (default 5) and `--repeats` (default 20). With no options all 15 datasets run. The
published April wall-clock time for all 15 datasets was about 10 hours on the author's machine
(a single recorded run, taken from the April run log, which is not included here). ESR and DCCC are
the largest datasets. ERF-CSRBoost fits 20 bags in every fold, so
even small datasets are slow. The run printed to the console only and kept nothing until a dataset
finished; an interrupted run leaves no result file. A smaller first check is a single small
dataset with fewer repeats (for example `--datasets BCW --repeats 2`). It was not run here and its
duration was not measured.

The commands above were **not executed** during release preparation.

## 6. Output

The runner writes only to `research/outputs/comparison_<YYYYmmdd_HHMMSS>/`:
`fold_results.csv` (columns `dataset, method, fold, acc, auc, f1, ap, gmean`) and
`dataset_summary.csv`. A search of the script for file writes finds only these two `to_csv` calls and
the creation of the `outputs` folders; everything else goes to the console. It does not write into
`results/` and does not touch the stored records. `outputs/` is ignored by Git.

**Behaviour on errors (read from the code, not executed).** Three places in the runner turn a failure
into something that does not stop the run:

- `_apply_smote` (lines 448 to 455) wraps the SMOTE call in `try`, catches any exception (line 454)
  and returns its input unchanged (line 455). A resampling failure therefore becomes a silent
  fallback to the retained rows without oversampling. It is called for CSRBoost (line 575) and for
  each ERF-CSRBoost bag (line 595). Nothing records whether the fallback was used, and no per-fold
  fallback record exists, so it was not established whether a fallback occurred in the stored June
  record.
- A dataset whose loader raises is printed as `SKIP -- load error` and skipped (lines 716 to 718); a
  key without a loader is skipped the same way (lines 707 to 710).
- A method that raises in a fold gets NaN for all five metrics (lines 740 to 742). The NaN values are
  written to `fold_results.csv`, the `ERROR` text is not, and the summary means leave the NaN values
  out (line 767, and the `np.nanmean` calls from line 774).

A run that finishes is therefore not proof of a complete experiment. After a run, count the rows for
each dataset and method (100 expected) and look for NaN. The stored June record passes both checks:
6,000 rows, 15 datasets x 4 methods x 100 folds (0 to 99), no NaN (recomputed).

## 7. Comparing a fresh run with the stored records

Do not expect bit-for-bit agreement: the library versions of the stored runs are unknown, and in the
author's own runs the small datasets matched exactly while ESR and DCCC differed in the fourth
decimal between the 28 April run and the June reruns, for a reason that was not established. Compare
dataset means (see the snippet in `../docs/RESULTS_EVIDENCE.md` section 10) against the June record,
which is the headline record and has the corrected CB rows, not against the April run.

## 8. The three result sets

**June four-way record (headline).** It is the April run with the rows of three datasets, ESR, DCCC
and CB, replaced by rows from June 2026 reruns that used the corrected loaders. Only CB changes
materially, because the positive class changed: in the April run the larger class (mines, 111 rows)
was the positive class; in the corrected run the smaller class (rocks, 97 rows) is. The twelve other
datasets are row for row identical in the two records (recomputed here). The two-method file that the
author kept with the thesis submission (CSRBoost and ERF-CSRBoost only, SHA-256
`7b1e353bba72c4221fa5c811624111fe2d1b805683934e4c6475ce0ef4532ff2`) equals the CSRBoost and
ERF-CSRBoost rows of this record, row for row (recomputed here), so it is not included separately.

Mean AUC over 100 folds, from the two records in `results/` (recomputed):

| Dataset | CSRBoost April | CSRBoost June | ERF-CSRBoost April | ERF-CSRBoost June |
|---|---|---|---|---|
| CB | 0.9028 | 0.9049 | 0.9362 | 0.9319 |
| ESR | 0.9776 | 0.9777 | 0.9960 | 0.9960 |
| DCCC | 0.7631 | 0.7630 | 0.7710 | 0.7710 |

In the June record ERF-CSRBoost has a higher mean AUC than CSRBoost on 12 of 15 datasets (mean
difference +0.0116, dataset-level Wilcoxon p 0.030), the same picture as April (+0.0120, p 0.030).
This is an unequal-capacity comparison (ERF-CSRBoost 1,000 trees, CSRBoost 50 stumps, the random
forest baselines 100 trees) on 15 datasets that were also used while the method was developed, so it
does not show a better method (`../docs/RESULTS_EVIDENCE.md`, sections 3.2, 3.4 and 4.3).
The June reruns of ESR and DCCC reproduce the 25 April runs of those datasets, not the 28 April
run (historical artifact, unverified here; the cause is unresolved).

**April run (earlier run).** Run `comparison_20260428_141315`. Its ESR, DCCC and CB rows were
superseded by the June reruns. The first public version of this repository used it as its
headline record.

**Exploratory 18-dataset run (supplement).** The published file is a filtered copy of a 9,000-row
file (SHA-256 `092857d64b228dd02f7fba697a7b7b7420cb6bbeed0e6e93a176004e80ddcb3f`). The original file
also contained rows for further models; they are not part of this release and were removed, because
this release analyses only CSRBoost, ERF and BRF. The filter keeps the header and every row whose
`model` field is `CSRBoost`, `ERF` or `BRF`, byte for byte and in the original order and column
order (5,400 rows remain; SHA-256
`560ef44c69abd862118237669f5d1c6a44a897ded571e46f4716d03bfe629a5f`). The `pick` column is empty
in every row of the published file and is not analysed. The run's script is not included, so the run cannot be reproduced from this
repository; these are historical raw fold records only. Its protocol differs from the headline
record (each method is trained on 70 percent of each training fold; the balanced random forest has
1,000 trees; five external datasets were added; ESR and DCCC were skipped). It is uncorrected and
exploratory. See `../docs/RESULTS_EVIDENCE.md`, section 8, for the analysis and every caveat.

## 9. Left out on purpose

Datasets; paper PDFs; Word and other reports; other working files, backups and old
workspaces; the April run log (a console transcript, not needed to recompute anything); the
per-dataset launcher scripts, the resumable wrapper and the older ERF-only runner; the older
snapshot of the runner (it cannot load ESR and DCCC and has the wrong Sonar polarity); the earlier
anchored-CSRBoost run and the aborted April attempt; the two-method thesis file (redundant, see
section 8); and the original files of the exploratory run and its script.
