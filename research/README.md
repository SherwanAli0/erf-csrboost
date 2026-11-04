# research/: runner script and stored results

ERF-CSRBoost is an experimental extension of CSRBoost and an unpublished research prototype. This
folder holds the script that produced the stored comparison and the saved per-fold records of that
comparison. It holds no datasets, no thesis or report documents, and no other research material.

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
| `results/comparison_20260428_141315/fold_results.csv` | The release record: 6,000 rows = 15 datasets x 4 methods x 100 folds, columns `dataset, method, fold, acc, auc, f1, ap, gmean` | CC BY 4.0 |
| `results/comparison_20260428_141315/dataset_summary.csv` | Per-dataset size, imbalance, and mean and standard deviation of each metric for each method | CC BY 4.0 |
| `results/june2026_two_method_rerun/fold_results.csv` | A later corrected two-method rerun (CSRBoost and ERF-CSRBoost only, 3,000 rows). It is included only because it supports the corrected CB numbers in section 6. It is not the release record | CC BY 4.0 |
| `MANIFEST.sha256` | SHA-256 of the four files above. Check with `cd research` then `sha256sum -c MANIFEST.sha256` (or `Get-FileHash` in PowerShell). `.gitattributes` forces LF line endings on this list so the check also works after a Windows checkout with `core.autocrlf=true` | MIT (configuration file) |
| `.gitignore` | Keeps downloaded datasets and the runner's own outputs out of Git | MIT (configuration file) |

All four files were copied byte for byte from the author's research working folder (a read-only
source) and compared by SHA-256 before and after copying, and again after the Git commit. They use
CRLF line endings. The repository's `.gitattributes` marks them `-text` so no checkout changes
their bytes. If a tool does convert them, the LF-normalised hashes in
`../docs/RESULTS_EVIDENCE.md` section 11 and `../docs/IMPLEMENTATION_MAP.md` apply instead.

The header comment of the runner is the author's original working text. It was left unchanged so
that the hash matches. It names the author's university and supervisor and describes ERF-CSRBoost
with the words "proposed novelty"; the current wording is in `../README.md`.

### How the runner relates to the stored April records

The April run (28 to 29 April 2026) was produced by an earlier version of this script. The exact
April text was not preserved, so its identity with this file is inferred, not proven. This file is the
later, corrected version: on 20 June 2026 the author fixed three loader problems (the ESR path, the
DCCC path, and the class polarity of the Sonar dataset, CB). The functions that implement
ERF-CSRBoost are the same in this file and in an older May 2026 snapshot of the script (compared by
syntax tree in a review pass, not by this repository). Consequences:

- Running this script on CB gives the corrected CB values, not the April ones (section 6).
- Running it on ESR and DCCC works; the older snapshot could not load those two datasets.
- The Python and library versions of the April run were not recorded.

## 2. What was and was not checked

Checked during release preparation (read-only, no training):

- SHA-256 of every file against the values in this repository's documents and `MANIFEST.sha256`.
- The runner compiles (syntax check only, in a scratch copy, with Python 3.11).
- The per-fold records recompute to the numbers in `../docs/RESULTS_EVIDENCE.md`: 6,000 rows with
  no missing values, the global means, ERF-CSRBoost higher than CSRBoost on 12 of 15 datasets, the
  dataset-level Wilcoxon p of 0.030 (Holm 0.060), the Friedman p of 0.005, and the four subset
  checks. The summary file's means recompute from the fold records. The June records recompute to
  the CB, ESR and DCCC values in section 6 and to 12 of 15 datasets.
- The runner's paths were read from the code (sections 3 and 5).

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
why they are link-only.

| Run key | Place the file at | Source | Positive class used by the loader (read from the code) |
|---|---|---|---|
| PSDAS | `01_PSDAS/data.csv` (semicolon separated, column `Target`) | UCI, Predict Students' Dropout and Academic Success | `Enrolled` |
| ESR | `02_ESR/Epileptic Seizure Recognition.csv` (the loader takes the first `*.csv` in the folder) | UCI, Epileptic Seizure Recognition (derived from University of Bonn recordings). Terms not established: do not redistribute | label 1 against the rest |
| DCCC | `03_DCCC/DCCC.xls` (first `*.xls` or `*.xlsx`; header on the second row; first column dropped) | UCI, Default of Credit Card Clients | default = 1 (flipped if it were the majority) |
| BCW | `04_BCW/wdbc.data` | UCI, Breast Cancer Wisconsin (Diagnostic) | `M` |
| ESDRP | `05_ESDRP/diabetes_data_upload.csv` (column `class`) | UCI, Early Stage Diabetes Risk Prediction | `Negative` (the smaller class) |
| CB | `06_CB/sonar.all-data` | UCI, Connectionist Bench (Sonar) | `R` (rocks, 97 rows; June and later runs) |
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
  per dataset and method.
- CB, ESDRP, FLAREF and others have properties (class polarity, duplicate rows) that matter for the
  results; see `../docs/RESULTS_EVIDENCE.md` section 3.

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

## 7. Comparing a fresh run with the stored records

Do not expect bit-for-bit agreement: the library versions of the stored run are unknown, and in the
author's own two runs the small datasets matched exactly while ESR and DCCC differed in the fourth
decimal, for a reason that was not established. Compare dataset means (see the snippet in
`../docs/RESULTS_EVIDENCE.md` section 10), and compare CB against the June records below, not the
April ones.

## 8. The June corrected records (CB, ESR, DCCC)

After April the author corrected the loader problems and re-ran ESR, DCCC and CB. Only CB changes
materially, because the positive class changed: in the April run the larger class (mines, 111 rows)
was the positive class; in the corrected run the smaller class (rocks, 97 rows) is. The twelve other
datasets are row for row identical in the two sets for CSRBoost and ERF-CSRBoost (recomputed here).

Mean AUC over 100 folds, from the two files in `results/` (recomputed):

| Dataset | CSRBoost April | CSRBoost June | ERF-CSRBoost April | ERF-CSRBoost June |
|---|---|---|---|---|
| CB | 0.9028 | 0.9049 | 0.9362 | 0.9319 |
| ESR | 0.9776 | 0.9777 | 0.9960 | 0.9960 |
| DCCC | 0.7631 | 0.7630 | 0.7710 | 0.7710 |

In the June two-method records ERF-CSRBoost has a higher mean AUC than CSRBoost on 12 of 15
datasets (mean difference +0.0116, dataset-level Wilcoxon p 0.030), the same picture as April. The
June file has no random forest baselines, so it says nothing about RF-BRF or RF-Raw. The April
four-way record is kept as the release record because only it contains the baselines.

## 9. Left out on purpose

Datasets; paper PDFs; Word and other reports; other working files, backups and old
workspaces; the April run log (a console transcript, not needed to recompute anything); the
per-dataset launcher scripts, the resumable wrapper and the older ERF-only runner; the older May
snapshot of the runner (it cannot load ESR and DCCC and has the wrong Sonar polarity); the earlier
anchored-CSRBoost run and the aborted April attempt; and the exploratory July variants (ERF-HB,
ERF-Auto, ExtraTrees, threshold calibration, 20 datasets), which are different methods and are not
part of this release. Only plain ERF-CSRBoost is described here.
