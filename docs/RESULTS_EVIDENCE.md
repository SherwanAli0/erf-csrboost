# Results evidence

What the stored ERF-CSRBoost results are, how they were produced, what was recomputed from the
saved records, and what cannot be concluded from them.

ERF-CSRBoost is an experimental extension of CSRBoost and an unpublished research prototype. The
numbers below are a historical artifact of an exploratory comparison on development datasets. They
are not an independent final evaluation and they do not establish a general improvement.

Version note: an earlier version of this document (first public commit, 4 November 2026) used the
April 2026 run as its headline record. This version uses the June four-way record, which is the
record behind the author's graduation thesis tables (the thesis itself is not part of this
repository), and keeps the April run as the earlier run. The substance is unchanged: ERF-CSRBoost
has a higher mean AUC than the re-implemented CSRBoost on 12 of 15 datasets in both (mean difference
+0.0120 in April, +0.0116 in June; dataset-level Wilcoxon p 0.0302 in both). That comparison has
unequal capacity (ERF-CSRBoost 1,000 trees, CSRBoost 50 stumps, the random forest baselines 100 trees)
and the 15 datasets were also used while the method was developed (sections 3.2 and 3.4), so it does
not show a better method. Global mean AUC moved from 0.9386 to 0.9383 for ERF-CSRBoost, from
0.9265 to 0.9267 for CSRBoost and from 0.9386 to 0.9389 for RF-BRF. It also adds an exploratory run on 18 datasets (section 8).

## 1. Status labels

| Label | Meaning |
|---|---|
| Historical artifact | A number or statement in a saved file or record that was read but not regenerated |
| Recomputed | Regenerated from the saved per-fold records (`fold_results.csv`) with Python, pandas, NumPy and SciPy; no model was trained. The short snippet in section 10 reproduces the main numbers |
| Newly executed check | A new, cheap check run during release preparation on saved data files, with no model training |
| Read from code | A person read the runner script |
| Unresolved | Could not be established from the artifacts available |
| Unverified here | Taken from the author's own records or notes that are not part of this repository, so this repository cannot confirm it |

## 2. The records

| Record | Role | Folder under `research/results/` | Rows |
|---|---|---|---|
| June four-way record | Headline record. 15 datasets x 4 methods x 100 folds; the record behind the graduation thesis tables | `comparison_june2026_four_way/` | 6,000 |
| April run `comparison_20260428_141315` | The earlier run (28 to 29 April 2026), which the first public version of this repository used as its headline. Its ESR, DCCC and CB rows were superseded by June reruns | `comparison_20260428_141315/` | 6,000 |
| Exploratory 18-dataset run | Supplement, not the headline. 18 datasets x 3 methods x 100 folds, a different protocol (section 8) | `exploratory_18_dataset_run/` | 5,400 |

Release artifacts and their SHA-256 are listed in section 11.

### 2.1 The June four-way record

| Item | Finding |
|---|---|
| Methods | CSRBoost (re-implemented, run live in every fold), ERF-CSRBoost, RF-Raw, RF-BRF |
| Datasets | 15 |
| Records | `fold_results.csv`: 15 datasets x 4 methods x 100 folds = 6,000 rows, folds 0 to 99 for every dataset and method, no missing values, no duplicate keys (recomputed) |
| How it was assembled | It is the April run with the rows of three datasets, ESR, DCCC and CB, replaced by rows from June 2026 reruns of those datasets (historical artifact). Recomputed here: the other 12 datasets are row for row identical in the two records, and the 3 reran datasets equal their June rerun files (those files are not part of this repository) |
| Summary files | `dataset_summary.csv` (per-dataset size, imbalance, and mean and standard deviation of each metric, population formula) and `global_means.csv` (mean of the 15 dataset means) are included. Both were recomputed from `fold_results.csv` and agree to within 3.3e-16. The record's own README is not included: it was not checked against the rows and is not used |
| Not recorded | Python and library versions of the runs; per-method timing; any flag showing whether the SMOTE fallback ever triggered |
| Thesis two-way file | The two-method file kept with the thesis submission (CSRBoost and ERF-CSRBoost only, 3,000 rows, SHA-256 `7b1e353bba72c4221fa5c811624111fe2d1b805683934e4c6475ce0ef4532ff2`) equals the CSRBoost and ERF-CSRBoost rows of this record, row for row (recomputed, text of every row). It is therefore not included separately |

**What changed from April to June (recomputed).** Mean AUC over 100 folds for the three datasets
whose rows were replaced:

| Dataset | CSRBoost April | CSRBoost June | ERF April | ERF June |
|---|---|---|---|---|
| CB | 0.9028 | 0.9049 | 0.9362 | 0.9319 |
| ESR | 0.9776 | 0.9777 | 0.9960 | 0.9960 |
| DCCC | 0.7631 | 0.7630 | 0.7710 | 0.7710 |

Number of the 100 folds whose five stored metrics differ between the April and June records:

| Dataset | CSRBoost | ERF-CSRBoost | RF-Raw | RF-BRF |
|---|---|---|---|---|
| CB | 100 | 100 | 100 | 100 |
| ESR | 5 | 59 | 0 | 0 |
| DCCC | 9 | 80 | 19 | 24 |

- **CB.** The positive class was corrected from the larger Sonar class (M, mines, 111 rows) in April to
  the smaller class (R, rocks, 97 rows) in June. Every CB row therefore differs. CB mean AP for the four
  methods (CSRBoost, ERF-CSRBoost, RF-Raw, RF-BRF) is 0.9146, 0.9494, 0.9396, 0.9379 in April and
  0.9013, 0.9278, 0.9226, 0.9225 in June.
- **ESR and DCCC.** The dataset-mean AUC of every method differs by at most 0.0001 between April and
  June. Some individual folds differ (table above); the largest single-fold difference in any stored
  metric is 0.0201 on DCCC and 0.0184 on ESR. The June reruns reproduce the 25 April runs of the same
  datasets (0 differing folds for ERF and CSRBoost), not the 28 April run (historical artifact,
  unverified here: the 25 April files are not part of this repository). The cause is unresolved, so
  the runner is not shown to be deterministic for these two large datasets. A plausible but untested
  explanation is a library or thread-count effect.
- The June reruns used the corrected runner included here (`research/scripts/`). It differs from a
  later snapshot of the earlier runner in three places only: the two folder names (ESR and DCCC), the
  Sonar positive class together with a guard that flips it if it would be the majority, and one
  docstring (a diff of the two files; historical artifact for the snapshot, see
  `IMPLEMENTATION_MAP.md`, section 3). The snapshot carries no reliable date: a comment in it refers
  to a reorganisation "after 2026-05-04", so it is from that day or later, and nothing else here
  dates it.

### 2.2 The April run (the earlier run)

| Item | Finding |
|---|---|
| Run identifier | `comparison_20260428_141315`. Started 28 April 2026 14:13:15 and finished about 29 April 2026 00:15; 36,102 s (10.0 hours) of wall clock according to the run log (historical artifact; the log is not part of this repository) |
| Records | `fold_results.csv` (6,000 rows, same layout as the June record) and `dataset_summary.csv` |
| Status | Earlier run. Its ESR, DCCC and CB rows were superseded by the June reruns (section 2.1). It is kept for provenance and because the first public version of this repository described it. Its other 12 datasets are identical to the June record |
| An aborted attempt | `comparison_20260428_141201` ran only one dataset for 40 rows and was superseded by the run above. It is not used |

A `global_means.csv` file from the April run is not included in this repository.

## 3. Datasets and protocol

### 3.1 The 15 datasets

Counts come from the dataset files themselves (checked by a review pass) and agree with
`dataset_summary.csv` of the June record for n, features, minority and majority counts (recomputed
for that file). "Label 1" is the positive class used for every metric in the June record. The file
hash column holds the first 12 hex characters of the SHA-256 of the data file used, so a reader can
check a downloaded copy.

| Key (run order) | Source of the file | n | Features | Label 1 means | Label 1 | Label 0 | Imbalance | File hash |
|---|---|---|---|---|---|---|---|---|
| PSDAS (1) | UCI | 4,424 | 36 | Enrolled (other two outcomes = 0) | 794 | 3,630 | 4.57 | `3ef126de5cef` |
| ESR (2) | UCI | 11,500 | 178 | class 1 (seizure) | 2,300 | 9,200 | 4.00 | `5ac55c71de5d` |
| DCCC (3) | UCI | 30,000 | 23 | default payment | 6,636 | 23,364 | 3.52 | `f61941714a35` |
| BCW (4) | UCI | 569 | 30 | M (malignant) | 212 | 357 | 1.68 | `d606af411f3e` |
| CB (5) | UCI | 208 | 60 | R (rocks), the smaller class (June record; the April run used M, mines, 111 rows) | 97 | 111 | 1.14 | `e90434cdbf00` |
| ESDRP (6) | UCI | 520 | 16 | class Negative (no diabetes) | 200 | 320 | 1.60 | `7889d9d0beb7` |
| ILPD (7) | UCI | 583 | 10 | class 2 (non-liver-patient in the UCI coding) | 167 | 416 | 2.49 | `8a3a32776f03` |
| GLASS (8) | UCI | 196 | 9 | type 7 | 29 | 167 | 5.76 | `d706f22a8b93` |
| SEED (9) | UCI | 210 | 7 | variety 1 | 70 | 140 | 2.00 | `1f3f83c0d848` |
| WINE (10) | UCI | 178 | 13 | cultivar 3 | 48 | 130 | 2.71 | `6be6b1203f3d` |
| YEAST5ERL (11) | KEEL | 1,484 | 8 | KEEL "positive" | 5 | 1,479 | 295.80 | `d9a250c05310` |
| CARGOOD (12) | KEEL | 1,728 | 6 | "good" | 69 | 1,659 | 24.04 | `96d72551a93d` |
| CARVGOOD (13) | KEEL | 1,728 | 6 | "very good" | 65 | 1,663 | 25.58 | `2ff0af74db2c` |
| YEAST5 (14) | KEEL | 1,484 | 8 | KEEL "positive" | 44 | 1,440 | 32.73 | `6c130fd581a6` |
| FLAREF (15) | KEEL | 1,066 | 11 | KEEL "positive" | 43 | 1,023 | 23.79 | `8c866c8cc80a` |

Notes that matter when reading the results:

- **CB (Sonar).** In the June record, label 1 is "R", the smaller class (97 rows), which is the
  corrected polarity. In the April run it was "M", the larger class (111 rows), so the resampling step
  treated the larger class as the minority and oversampled the smaller one (read from code, and from
  the class counts). AUC does not change when the positive class is swapped for the same scores, but
  the balancing step, AP, F1 and G-Mean do (section 2.1).
- **ESDRP.** 269 of 520 rows are exact duplicates of other rows (features and class), counted from the
  file. A review pass found that, on average, 68.7 percent of test rows in the run's own folds have an
  exact twin in the training fold (newly executed check by that pass, not repeated here). Near-1.0 AUC
  for every method on ESDRP, and ERF's gain over CSRBoost there, should be read with that in mind.
- **FLARE-F.** 760 of 1,066 rows are exact duplicates of other rows (counted from the file). On average
  80.6 percent of test rows have an exact twin in the training fold (the same review pass). This is
  where ERF-CSRBoost loses to CSRBoost and RF-BRF. Duplicates plausibly affect every method, but the
  pattern is not explained.
- **YEAST5 and YEAST5ERL.** 31 duplicate rows each. YEAST5ERL has only 5 positives, so each stratified
  test fold holds exactly one positive; AUC, AP and G-Mean are then computed from a single positive,
  and F1 is 0 or 1 in every fold. The two files also hold the same 1,484 feature rows (section 3.5),
  so they are not independent datasets either.
- **GLASS.** The file has 196 rows (types 1, 2, 5, 6 and 7 only). The original UCI glass data have 214
  rows. Whether this 196-row file is the glass dataset used by the CSRBoost paper is unresolved. The
  row counts and the comparability consequence are in section 3.5.
- **CARGOOD and CARVGOOD** share identical feature rows and differ only in which class is positive,
  so they are not independent datasets. The effective number of independent datasets is below 15.
- **Exact feature twins and shared feature collections.** Section 3.5 gives, for six datasets, how many
  test rows of the first fold have a feature twin in training, and lists the task pairs that share
  their whole feature set (CARGOOD with CARVGOOD, and YEAST5 with YEAST5ERL).
- Several datasets are multi-class files collapsed to a binary task: PSDAS, ESR, SEED, WINE, GLASS.
- The statement "all 15 datasets are the ones listed in the CSRBoost paper" comes from the research
  files. It was not checked against the paper's table.
- Loader steps applied to the whole data before cross-validation use no label-dependent statistic:
  coercion to numeric with missing values set to 0, label encoding of text columns by sorted value,
  and fixed ordinal maps for the car and flare files. The scaler, clustering and SMOTE are fitted
  inside each training fold (read from code).

### 3.2 Methods and capacity

| Method | Configuration (read from code) | Capacity per fold |
|---|---|---|
| CSRBoost (re-implemented, live) | MiniBatchKMeans (n_init 1, max_iter 100, batch 1024) with K = minority count, keep 50 percent per cluster, SMOTE k = min(5, n_min - 1), AdaBoost of 50 depth-1 stumps. One draw per fold, seed 42 for clustering, SMOTE and AdaBoost | 50 stumps |
| ERF-CSRBoost | 20 bags. Each bag: the same cluster, 50 percent and SMOTE steps with seed 42 + 7919 b, then a RandomForest of 50 trees. Test probabilities averaged over bags | 20 x 50 = 1,000 fully grown trees, plus 20 clusterings and 20 SMOTE passes |
| RF-Raw | RandomForest, 100 trees, sqrt features, fully grown, no class weights, seed 42 | 100 trees |
| RF-BRF | BalancedRandomForest, 100 trees, sampling_strategy "all", replacement True, bootstrap False, seed 42 | 100 trees, each on 2 x n_min rows |

**Unequal capacity.** ERF-CSRBoost uses 10 times the trees of the two random forest baselines and
20 times the clustering and resampling work of CSRBoost. Its probability output is an average of 20
differently resampled forests, while CSRBoost is a single stochastic draw of 50 stumps. The
baselines are untuned library defaults. No equal-capacity baseline (for example a 1,000-tree random
forest) exists in the 15-dataset record. The comparison uses the same folds and the same metrics for
all four methods; it does not use the same compute budget. The one comparison against a 1,000-tree
balanced random forest is in the exploratory run (section 8), under a different protocol.

### 3.3 Evaluation protocol (read from code, fold structure recomputed)

| Element | Finding |
|---|---|
| Split | RepeatedStratifiedKFold, 5 splits, 20 repeats, random_state 42: 100 train and test folds per dataset. All four methods see identical folds within a dataset (each method has fold indices 0 to 99 for every dataset) |
| Seeds | Splits 42. CSRBoost: 42 in every fold. ERF: 42 + 7919 b in every fold. RF-Raw and RF-BRF: 42 in every fold. The random components are therefore re-used identically across folds |
| Preprocessing | StandardScaler fitted on the training fold only; clustering and SMOTE fitted on the scaled training fold only; no resampling of test rows |
| SMOTE failure | `_apply_smote` (runner lines 448 to 455) catches any exception (line 454) and returns its input unchanged (line 455), so an exception makes the runner silently use the retained rows without oversampling. It is called for CSRBoost (line 575) and for every ERF bag (line 595). The run does not record whether this ever happened, and no per-fold fallback record exists, so it is not established whether a fallback occurred in the stored June record |
| Other failures | A dataset whose loader raises is printed as `SKIP -- load error` and skipped (lines 716 to 718). A method that raises in a fold gets NaN for every metric (lines 740 to 742), and the summary means leave the NaN values out (line 767 and the `np.nanmean` calls from line 774). The NaN values are written to `fold_results.csv`, but the `ERROR` text is not. A finished run is therefore not proof of a complete experiment; the expected rows must be counted and checked for NaN. The stored June record has all of them: 6,000 rows, 15 datasets x 4 methods x 100 folds (0 to 99), no NaN (recomputed) |
| Threshold | Fixed 0.5 for ACC, F1 and G-Mean. AUC and AP come from probabilities. There is no threshold selection |
| Hyperparameters | No search in this run. All settings are fixed constants |
| Aggregation | Per-fold metrics on the test fold; dataset value = unweighted mean over its 100 folds; global value = unweighted mean of the 15 dataset values (equal to the mean over all 1,500 pooled rows per method because each dataset has exactly 100 folds) |

### 3.4 Exploratory, not independent

Inside the run there is no use of held-out labels for selection. Around the run, the same 15
datasets were used throughout development: earlier ERF runs, comparison runs on several April days,
loader fixes for some datasets, replacement of paper-reported baseline values by a live baseline,
and earlier method candidates on the same data. The selection effect of that history cannot be
quantified from the records. The four-way table is therefore an exploratory, development-set
comparison with an untuned fixed configuration, not an independent final evaluation.

### 3.5 Data quality: feature twins, shared feature collections and the GLASS file version

These checks load the data files with the runner's loader logic and look at the feature matrices in
memory. No model was fitted, the runner was not run, and no stored number changes. The check script
is a scratch script that is not in this repository. Status of every figure in this section: **newly
executed check** (recomputed from the data files).

**Exact feature twins across train and test (one split).** A test row is counted when its full feature
vector, with the class label ignored, also occurs among the training rows. The counts below are for
the first fold only (fold 0 of `fold_results.csv`: the first split of `RepeatedStratifiedKFold` with
5 splits, 20 repeats and `random_state` 42, on the loaded feature matrix before scaling). This is a
diagnostic for one split, not an average over the 100 evaluations.

| Dataset | Rows | Features | Rows that repeat an earlier feature vector | Test rows with a feature twin in training | Test rows in that fold | Share |
|---|---|---|---|---|---|---|
| ESDRP | 520 | 16 | 269 | 79 | 104 | 76.0% |
| FLARE-F | 1,066 | 11 | 779 | 177 | 214 | 82.7% |
| DCCC | 30,000 | 23 | 56 | 20 | 6,000 | 0.3% |
| ILPD | 583 | 10 | 13 | 4 | 117 | 3.4% |
| YEAST5 | 1,484 | 8 | 31 | 7 | 297 | 2.4% |
| YEAST5ERL | 1,484 | 8 | 31 | 7 | 297 | 2.4% |

- **Relation to section 3.1.** The 269 and 760 duplicate rows of ESDRP and FLARE-F in section 3.1
  count a row as a duplicate only if its class also matches. Ignoring the class, 779 FLARE-F rows
  repeat an earlier feature vector (ESDRP stays at 269). The review-pass averages over all 100 folds
  in section 3.1 (68.7 and 80.6 percent) agree with the average share of test rows that have a
  training twin of the same class, recomputed here as 68.65 percent for ESDRP and 80.64 percent for
  FLARE-F; ignoring the class, the FLARE-F average over the 100 folds is 82.74 percent. In the first
  fold, 171 of the 177 FLARE-F twins and all 79 ESDRP twins have a training twin of the same class.
- **Conflicting labels.** FLARE-F has 19 feature patterns that occur with both classes, and DCCC has
  21. ESDRP, ILPD, YEAST5 and YEAST5ERL have none.
- **Exact-feature overlap is not resampling before splitting.** The runner fits the scaler on the
  training fold and passes only training rows to clustering and SMOTE (section 3.3), so no learned step
  sees a test row. Feature twins are a property of the original data: a test row can still have an
  identical copy among the training rows, and the score on such a row can reflect memorisation of
  the training copy rather than generalisation. This is why duplicates can inflate test scores
  (section 7, item 4).
- **The saved results were not deduplicated.** Every stored number includes the twins. Removing or
  grouping them would change the experiment, not repair a value, so nothing was done here. A study of
  generalisation would have to choose a duplicate or group policy first. The exploratory 18-dataset
  run (section 8) uses the same loaders for 13 datasets, so the same twins are present; the five added
  datasets were not checked.

**Tasks that share a whole feature collection.** Compared as sorted feature rows after loading with
the runner's loader logic:

| Task pair | Rows | Features | Same rows? | Positives |
|---|---|---|---|---|
| CARGOOD and CARVGOOD | 1,728 | 6 | Yes, same order | 69 and 65 |
| YEAST5 and YEAST5ERL | 1,484 | 8 | Yes, file order differs | 44 and 5 |

The 15 task labels are therefore not 15 separate feature collections: two pairs differ only in the
target label. No other two tasks have the same number of rows (section 3.1), so these are the only
candidate pairs. The dataset-level tests of sections 4.3 and 8.3 still treat the tasks as
independent, so the effective number of independent datasets is below 15 (and below 18 in section 8).
Section 4.3 shows a sensitivity check without CARVGOOD; the equivalent check without one member of
the YEAST5 pair was not computed.

**GLASS file version.** The selected `glass.csv` has 196 rows and nine features: type 1 has 69 rows,
type 2 76, type 5 13, type 6 9 and type 7 29 (class counts recomputed from the file). Type 3 is absent.
A canonical 214-row file of the UCI glass data exists in the author's research folder (it is not part of this
repository): there type 1 has 70 rows, type 3 has 17 and the other counts are the same (read from
that file). The positive class used here (type 7) has 29 rows in both files, but the negative side
has 167 rows in the selected file and 185 in the canonical one. A rerun on the 214-row file would
therefore be a different experiment, not a reproduction of the stored GLASS rows. Record the exact file hash (section 3.1) when comparing GLASS results.

## 4. Recomputed results (June four-way record)

All numbers in this section are recomputed from the June `fold_results.csv`. Metrics are unweighted
means over 100 folds per dataset. Fixed 0.5 threshold, untuned baselines, unequal capacity (section
3.2), development datasets (section 3.4).

### 4.1 Mean AUC by dataset

| Dataset | CSRBoost | ERF-CSRBoost | RF-Raw | RF-BRF | ERF minus CSRBoost | ERF minus RF-BRF | ERF minus RF-Raw |
|---|---|---|---|---|---|---|---|
| PSDAS | 0.7601 | 0.8212 | 0.8081 | 0.8134 | +0.0611 | +0.0078 | +0.0131 |
| ESR | 0.9777 | 0.9960 | 0.9952 | 0.9946 | +0.0183 | +0.0014 | +0.0007 |
| DCCC | 0.7630 | 0.7710 | 0.7637 | 0.7727 | +0.0080 | -0.0017 | +0.0073 |
| BCW | 0.9922 | 0.9911 | 0.9897 | 0.9906 | -0.0011 | +0.0005 | +0.0014 |
| CB | 0.9049 | 0.9319 | 0.9276 | 0.9278 | +0.0270 | +0.0040 | +0.0043 |
| ESDRP (duplicates) | 0.9753 | 0.9984 | 0.9985 | 0.9983 | +0.0231 | +0.0001 | -0.0001 |
| ILPD | 0.7276 | 0.7537 | 0.7433 | 0.7511 | +0.0261 | +0.0027 | +0.0104 |
| GLASS (196 rows) | 0.9514 | 0.9759 | 0.9755 | 0.9741 | +0.0245 | +0.0018 | +0.0004 |
| SEED | 0.9833 | 0.9802 | 0.9795 | 0.9775 | -0.0031 | +0.0027 | +0.0007 |
| WINE | 0.9993 | 1.0000 | 1.0000 | 0.9998 | +0.0007 | +0.0002 | +0.0000 |
| YEAST5ERL (5 positives) | 0.9996 | 1.0000 | 1.0000 | 0.9992 | +0.0004 | +0.0008 | +0.0000 |
| CARGOOD | 0.9760 | 0.9933 | 0.9954 | 0.9958 | +0.0172 | -0.0025 | -0.0021 |
| CARVGOOD | 0.9972 | 0.9999 | 0.9999 | 0.9998 | +0.0027 | +0.0001 | +0.0000 |
| YEAST5 | 0.9853 | 0.9912 | 0.9879 | 0.9913 | +0.0059 | -0.0001 | +0.0034 |
| FLAREF (duplicates) | 0.9073 | 0.8705 | 0.8378 | 0.8972 | -0.0368 | -0.0267 | +0.0327 |

### 4.2 Global means (mean of the 15 dataset means)

| Method | ACC | AUC | F1 | AP | G-Mean |
|---|---|---|---|---|---|
| CSRBoost (live) | 0.8955 | 0.9267 | 0.7248 | 0.7618 | 0.8707 |
| ERF-CSRBoost | 0.9188 | 0.9383 | 0.7558 | 0.8151 | 0.8441 |
| RF-Raw | 0.9261 | 0.9335 | 0.6749 | 0.8158 | 0.7256 |
| RF-BRF | 0.9016 | 0.9389 | 0.6952 | 0.8103 | 0.8918 |

At full precision ERF-CSRBoost is 0.938281 and RF-BRF is 0.938881, so RF-BRF is higher by 0.0006.
The two agree to three decimals. Accuracy, F1 and G-Mean at a fixed 0.5 threshold depend on how each
method calibrates probabilities and should not be read as ranking evidence.

### 4.3 Inference with the dataset as the unit (N = 15, AUC)

Folds of one dataset share training data and are positively correlated, so a test across all 1,500
pooled fold values overstates significance and is not used. The unit here is the dataset.

| Pair (unit = dataset, N = 15) | Mean difference | Datasets higher / lower | Wilcoxon signed-rank, two-sided p | Holm-adjusted p (family of three ERF comparisons) |
|---|---|---|---|---|
| ERF-CSRBoost minus CSRBoost | +0.0116 | 12 / 3 | 0.0302 | 0.0603 |
| ERF-CSRBoost minus RF-BRF | -0.0006 | 11 / 4 | 0.1514 | 0.1514 |
| ERF-CSRBoost minus RF-Raw | +0.0048 | 11 / 2 (2 ties) | 0.0107 | 0.0322 |

The RF-Raw p-value uses SciPy's normal approximation because two differences (WINE and YEAST5ERL)
are exactly zero; with the two zeros dropped the exact p is 0.0081.

- A Friedman test over the four methods on the 15 dataset AUCs gives chi-square 12.69, p = 0.0054.
  Average ranks (1 is best): ERF 1.67, RF-BRF 2.47, RF-Raw 2.53, CSRBoost 3.33. No post-hoc rank test
  (for example Nemenyi) was computed.
- A percentile bootstrap interval for the ERF minus CSRBoost mean AUC difference is +0.0010 to
  +0.0221 (2.5th and 97.5th percentile of 10,000 resampled means). Recipe: the 15 dataset differences
  in alphabetical order of the dataset key (BCW, CARGOOD, CARVGOOD, CB, DCCC, ESDRP, ESR, FLAREF,
  GLASS, ILPD, PSDAS, SEED, WINE, YEAST5, YEAST5ERL), `rng = numpy.random.default_rng(0)`, one index
  matrix `rng.integers(0, 15, size=(10000, 15))`, the mean of each resampled row. The snippet in
  section 10 prints it. With a fixed seed the endpoints still depend on the order of the 15 values in
  the array: in the order of the run keys (PSDAS, ESR, DCCC, ..., FLAREF) the same recipe gives
  +0.0014 to +0.0222. Either way the interval excludes zero only by a small margin.
- The Holm family (three comparisons of ERF against each other method) was chosen after seeing the
  data. All six pairwise comparisons, Wilcoxon p and Holm-adjusted p:

| Pair | Wilcoxon p | Holm-adjusted p (family of six) |
|---|---|---|
| CSRBoost vs ERF-CSRBoost | 0.0302 | 0.1206 |
| CSRBoost vs RF-Raw | 0.0554 | 0.1661 |
| CSRBoost vs RF-BRF | 0.0125 | 0.0645 |
| ERF-CSRBoost vs RF-Raw | 0.0107 | 0.0645 |
| ERF-CSRBoost vs RF-BRF | 0.1514 | 0.3029 |
| RF-Raw vs RF-BRF | 0.2293 | 0.3029 |

- Read every p-value in this section with three further limits. The models have unequal capacity
  (1,000 trees for ERF, 50 stumps for CSRBoost, 100 trees for RF-Raw and RF-BRF; no equal-capacity
  comparison was run in this record). The 15 datasets were also used while the method was developed,
  so they are not a fresh test set. The unit is the dataset, with N = 15, and the datasets are not
  independent (CARGOOD and CARVGOOD share features, and so do YEAST5 and YEAST5ERL; some have duplicate
  rows and test rows with exact feature twins in training, section 3.5).
- The result is borderline and not robust to removing questionable datasets:

| Subset | ERF minus CSRBoost mean difference | Datasets higher | Wilcoxon p |
|---|---|---|---|
| All 15 | +0.0116 | 12 / 15 | 0.0302 |
| Without YEAST5ERL (5 positives) | +0.0124 | 11 / 14 | 0.0353 |
| Without CB | +0.0105 | 11 / 14 | 0.0494 |
| Without ESDRP (duplicates) | +0.0108 | 11 / 14 | 0.0494 |
| Without YEAST5ERL and CB | +0.0113 | 10 / 13 | 0.0574 |
| Without CARVGOOD (shares features with CARGOOD) | +0.0122 | 11 / 14 | 0.0419 |

- CARGOOD and CARVGOOD share features, and so do YEAST5 and YEAST5ERL (section 3.5), so N = 15
  overstates independence.

### 4.4 Per-dataset paired results over the 100 folds

The corrected resampled t test of Nadeau and Bengio (variance factor 1/100 + 1/4, 99 degrees of
freedom) is used instead of a raw test over folds, with Holm adjustment across the 15 datasets. The
label rule is ours, chosen after seeing the data: "ERF better" or "ERF worse" only if the
Holm-adjusted corrected p is below 0.05 and the absolute mean difference is at least 0.005;
otherwise "tie or inconclusive".

ERF-CSRBoost minus CSRBoost, AUC:

| Dataset | Mean difference | Share of folds with ERF higher | Holm-adjusted corrected p | Label |
|---|---|---|---|---|
| PSDAS | +0.0611 | 1.00 | below 0.0001 | ERF better |
| ESR | +0.0183 | 1.00 | below 0.0001 | ERF better |
| DCCC | +0.0080 | 0.98 | 0.0086 | ERF better |
| BCW | -0.0011 | 0.44 | 1.0000 | tie or inconclusive |
| CB | +0.0270 | 0.79 | 0.9385 | tie or inconclusive |
| ESDRP | +0.0231 | 0.99 | 0.0044 | ERF better (duplicate-driven) |
| ILPD | +0.0261 | 0.76 | 0.7518 | tie or inconclusive |
| GLASS | +0.0245 | 0.56 | 1.0000 | tie or inconclusive |
| SEED | -0.0031 | 0.31 | 1.0000 | tie or inconclusive |
| WINE | +0.0007 | 0.10 | 1.0000 | tie or inconclusive |
| YEAST5ERL | +0.0004 | 0.11 | 1.0000 | tie or inconclusive |
| CARGOOD | +0.0172 | 1.00 | below 0.0001 | ERF better |
| CARVGOOD | +0.0027 | 0.98 | 0.0503 | tie or inconclusive |
| YEAST5 | +0.0059 | 0.88 | 0.7518 | tie or inconclusive |
| FLAREF | -0.0368 | 0.11 | 0.3046 | tie or inconclusive (probable loss) |

Under the same rule, ERF against RF-BRF is "tie or inconclusive" on all 15 datasets, and ERF against
RF-Raw is "ERF better" on 2 datasets (PSDAS and DCCC) and inconclusive on the other 13 (recomputed).

### 4.5 Other metrics, ERF-CSRBoost minus CSRBoost (unit = dataset, not corrected across metrics)

| Metric | Mean difference | Datasets higher / lower | Wilcoxon p (uncorrected) |
|---|---|---|---|
| AP | +0.0533 | 12 / 3 | 0.0054 |
| F1 | +0.0310 | 10 / 5 | 0.1070 |
| G-Mean | -0.0267 | 9 / 6 | 0.7615 |
| ACC | +0.0234 | 12 / 3 | 0.0020 |

G-Mean against RF-BRF, same test (recomputed): mean difference -0.0478, ERF higher on 9 datasets and
lower on 6, exact Wilcoxon p = 0.4543 (no zero differences). Neither G-Mean test is significant at the
dataset level (p 0.7615 against CSRBoost, 0.4543 against RF-BRF).

Two datasets, YEAST5ERL (one positive per test fold) and FLAREF, have by far the largest G-Mean
losses against CSRBoost (-0.329 and -0.223). Without them the mean G-Mean difference ERF minus CSRBoost is +0.0117 (13 datasets, 9
higher and 4 lower), so the lower global mean against CSRBoost is driven by those two datasets. The
mean difference ERF minus RF-BRF stays at -0.0038 without them (9 higher and 4 lower), so the lower
global mean against RF-BRF is largely, but not wholly, due to them. There is no evidence that
ERF-CSRBoost helps G-Mean at the 0.5 threshold.

The AP gain over CSRBoost is dominated by CARGOOD (CSRBoost AP 0.488, ERF 0.843; the dataset
difference is about +0.355); without CARGOOD the mean AP difference is about +0.032.

## 5. Where ERF-CSRBoost scores higher, ties and loses

Scope: 15 datasets, one protocol, fixed 0.5 threshold, untuned baselines, unequal capacity,
development datasets. Statements are about mean AUC unless a metric is named.

**Higher AUC than CSRBoost (not evidence of a better method).** Clear gains with fold-level agreement
on PSDAS (+0.061), ESR (+0.018), DCCC (+0.008) and CARGOOD (+0.017); ESDRP (+0.023) also passes the
rule but is duplicate-driven. Across datasets the mean AUC is higher on 12 of 15 and lower on 3 (BCW,
SEED and FLAREF; no ties) (mean +0.0116, Wilcoxon p 0.030, Holm 0.060 over a family chosen after
seeing the data and 0.121 over all six pairs, Friedman p 0.005). The capacity, development-set
and dataset-level limits in section 4.3 apply. Against RF-Raw the mean AUC is +0.0048 (11 of the 13
non-tied datasets), which is partly a capacity difference (1,000 trees against 100).

**Ties.** Against RF-BRF there is no detectable difference (mean -0.0006, p 0.151). On saturated
datasets (every method at an AUC of about 0.97 or above: BCW, ESDRP, SEED, WINE, YEAST5ERL, CARVGOOD,
and GLASS at about 0.95 to 0.98) ERF cannot be separated from the two random forest baselines (absolute
differences of at most 0.0027), and against CSRBoost the differences on BCW (-0.0011), SEED (-0.0031),
WINE (+0.0007), YEAST5ERL (+0.0004) and CARVGOOD (+0.0027) are in the third or fourth decimal. The
12 of 15 count therefore includes near-ties on saturated datasets (WINE, YEAST5ERL and CARVGOOD, where
ERF counts as higher by less than 0.005), so read the count together with the table in section 4.1.
The number of datasets where ERF is higher by at least 0.005 is 9 (PSDAS, ESR, DCCC, CB, ESDRP, ILPD,
GLASS, CARGOOD, YEAST5). ERF is higher by +0.006 to +0.027 on ILPD, CB, GLASS and YEAST5 against
CSRBoost, but not significantly under the corrected test.

**Loses or probably loses.** FLAREF: ERF is lower than CSRBoost by 0.0368 (higher in only 11 of 100
folds, corrected Holm p 0.30) and lower than RF-BRF by 0.0267; it is also the dataset with the most
duplicate rows. BCW (-0.0011) and SEED (-0.0031) favour CSRBoost by margins that are not meaningful.
DCCC, CARGOOD and YEAST5 against RF-BRF favour BRF by small margins. G-Mean at 0.5: ERF is lower than
CSRBoost (-0.0267) and RF-BRF (-0.0478) on the global mean. YEAST5ERL and FLAREF drive the CSRBoost
gap (without them +0.0117) but explain only most of the RF-BRF gap (without them -0.0038). The
dataset-level tests are not significant (p 0.7615 and 0.4543, section 4.5).

## 6. Status of each claim made in this repository

| Claim | Status | Basis |
|---|---|---|
| The June record has 6,000 per-fold rows (15 x 4 x 100) with no missing values | Recomputed | June `fold_results.csv` |
| Global means in section 4.2 | Recomputed | June `fold_results.csv`; the included `global_means.csv` agrees |
| ERF-CSRBoost has a higher mean AUC than the re-implemented CSRBoost on 12 of 15 datasets | Recomputed. Unequal capacity (1,000 trees against 50 stumps) and datasets that were also used during development: not evidence of a better method | sections 3.2, 3.4 and 4.3 |
| Wilcoxon p 0.030, Holm 0.060, Friedman p 0.005 | Recomputed. The Holm family is a post-hoc choice, and the result is borderline; the capacity (1,000 trees against 50 stumps and 100 trees) and development-set limits apply | sections 3.2, 3.4 and 4.3 |
| ERF-CSRBoost is not distinguishable from RF-BRF on AUC | Recomputed | section 4.3 |
| ERF-CSRBoost has a lower mean G-Mean than CSRBoost and RF-BRF at 0.5 | Recomputed (dataset-level Wilcoxon p 0.7615 against CSRBoost and 0.4543 against RF-BRF, not significant) | sections 4.2 and 4.5 |
| Five datasets show a clear ERF-over-CSRBoost gain | Recomputed, under a rule chosen after seeing the data; unequal capacity (1,000 trees against 50 stumps) and development datasets | section 4.4 |
| The protocol is 5-fold stratified, 20 repeats, same folds for every method, scaler on the training fold, threshold 0.5 | Read from code; fold structure recomputed | section 3.3 |
| No threshold selection or tuning inside the 15-dataset record | Read from code | section 3.3 |
| ERF uses 1,000 trees against 50 stumps and 100 trees | Read from code | section 3.2 |
| In the June record CB's positive class is the smaller class (R); in the April run it was the larger class (M) | Read from code and class counts | sections 2.1 and 3.1 |
| The June record equals the April run except for the ESR, DCCC and CB rows | Recomputed | section 2.1 |
| The thesis two-way file equals the CSRBoost and ERF-CSRBoost rows of the June record | Recomputed | section 2.1 |
| ESDRP and FLARE-F contain 269 and 760 duplicate rows | Newly executed check (counted from the files) | section 3.1 |
| 68.7 percent (ESDRP) and 80.6 percent (FLARE-F) of test rows have a training twin | Newly executed check by a review pass; recomputed in section 3.5 (68.65 and 80.64 percent for twins of the same class) | sections 3.1 and 3.5 |
| First fold: 79 of 104 ESDRP, 177 of 214 FLARE-F, 20 of 6,000 DCCC, 4 of 117 ILPD, 7 of 297 YEAST5 and 7 of 297 YEAST5ERL test rows have a feature twin in training; FLARE-F has 19 and DCCC 21 feature patterns with conflicting labels | Newly executed check (loaders only, no model fitted); a one-split diagnostic, not an average. The saved results were not deduplicated | section 3.5 |
| CARGOOD and CARVGOOD, and YEAST5 and YEAST5ERL, each share an identical set of feature rows | Newly executed check (sorted feature rows compared). Dataset-level tests still treat the tasks as independent | section 3.5 |
| The selected GLASS file has 196 rows (type 3 absent, type 1 has 69); the original UCI file has 214 | Newly executed check on the selected file; the 214-row counts were read from the canonical file in the author's research folder, which is not part of this repository. A rerun on 214 rows would not be the same experiment | section 3.5 |
| A SMOTE exception makes the runner fall back silently; load errors are skipped and metric errors become NaN | Read from code (lines 448 to 455, 716 to 718, 740 to 742). Whether a fallback occurred in the stored June record is not established; the record has all 6,000 expected rows and no NaN (recomputed) | section 3.3 |
| The June ESR and DCCC reruns reproduce the 25 April runs, not the 28 April run | Historical artifact, unverified here; cause unresolved | section 2.1 |
| The April run took about 10 hours | Historical artifact (run log); not checkable from this repository | section 2.2 |
| Library versions of the runs | Unresolved: not recorded | section 2.1 |
| In the exploratory run ERF-CSRBoost has higher AP than CSRBoost on 16 of 18 datasets, and no AUC or AP advantage over a 1,000-tree balanced random forest | Recomputed from the published raw records; exploratory, uncorrected; the CSRBoost comparison has unequal capacity (1,000 trees against 50 stumps) and 13 of the 18 datasets are development datasets | section 8 |
| The exploratory run can be reproduced from this repository | Not the case: its script is not included | section 8 |
| The 15 datasets are those of the CSRBoost paper | Unresolved: not checked against the paper; GLASS differs in row count from the original | section 3.1 |
| Why ERF scores higher where it does (mechanism) | Unresolved and untested: no ablation separates the classifier change, bagging and the larger number of trees | not claimed |
| That ERF helps more at moderate imbalance, or RF-BRF at extreme imbalance | Unresolved and untested | not claimed |
| The application reproduces these results | Not the case. The app is adapted from the runner and uses a different protocol | `IMPLEMENTATION_MAP.md` |

## 7. Limitations to keep next to the tables

1. The datasets were reused throughout development. This is not an independent evaluation.
2. No equal-capacity baseline in the 15-dataset record: 1,000 trees against 50 stumps and 100 trees.
   The only tree-count-matched comparison is the exploratory run's 1,000-tree balanced random forest
   (section 8), under a different protocol, and it shows no advantage for ERF-CSRBoost.
3. Fixed 0.5 threshold. Accuracy, F1 and G-Mean depend on calibration.
4. Duplicate rows (ESDRP, FLARE-F, and 31 in each YEAST5 file) can inflate test scores. In the first
   fold, 79 of 104 ESDRP test rows and 177 of 214 FLARE-F test rows have an exact feature twin in the
   training rows (section 3.5, a one-split diagnostic). The saved results were not deduplicated.
5. CB: the positive class was corrected from the larger to the smaller class in the June record. The
   April run (earlier run) used the larger class.
6. The GLASS file has 196 rows, not the 214 of the original UCI data. A rerun on the 214-row file
   would be a different experiment (section 3.5).
7. CARGOOD and CARVGOOD share their whole feature set, and so do YEAST5 and YEAST5ERL. YEAST5ERL has
   5 positives. The effective number of independent datasets is below 15.
8. The June record's ESR and DCCC rows agree with the 25 April runs and not with the 28 April run of
   the same datasets, for a reason that was not established. Bit-for-bit reproduction is not promised.
9. Python and library versions of the runs were not recorded. Small datasets reproduce exactly across
   the April and June records. Per-method timing was not recorded.
10. The comparison method labelled CSRBoost is a local re-implementation (MiniBatchKMeans, depth-1
    stumps, no class-redefinition guard). It is not the authors' code and may not reproduce the
    paper's own numbers. The separate audit repository studies that question for the published method.
11. The stored runner copy can load ESR and DCCC when the data folders have the layout in
    `research/README.md`; an earlier snapshot of the runner could not.

## 8. Extended exploratory run (18 datasets)

This is an **exploratory supplement, not the headline record**. It is a historical artifact: the raw
per-fold records of one run that the author made after the thesis record, with a different protocol
and 5 datasets that are not in the 15-dataset record. Everything in this section is recomputed from
the published records; nothing was retrained. Read it with the caveats in section 8.4.

### 8.1 What is published and what is not

| Item | Finding |
|---|---|
| Published file | `research/results/exploratory_18_dataset_run/fold_results.csv`: 5,400 rows = 18 datasets x 3 methods (CSRBoost, ERF, BRF) x 100 folds. Columns `dataset, src, fold, model, auc, ap, gmean_default, gmean_moved, pick` |
| Original file | 9,000 rows, SHA-256 `092857d64b228dd02f7fba697a7b7b7420cb6bbeed0e6e93a176004e80ddcb3f` |
| Filter rule | The original file also contained rows for further models. They are not part of this release and were removed; this release analyses only CSRBoost, ERF and BRF. The rule keeps the header and every row whose `model` field is exactly `CSRBoost`, `ERF` or `BRF`. Kept rows are byte for byte the original lines, in the original order and with the original column order. 9,000 rows before, 5,400 after. The `pick` column is empty in every row of the published file and is not analysed. The filter is deterministic; the SHA-256 of the published file is in section 11 |
| What the columns mean | `src` is the group label written by the run (`paper` for 13 of the 15 datasets of section 3.1, other labels for the added datasets). `gmean_default` is G-Mean at the fixed 0.5 threshold; `gmean_moved` is G-Mean at a threshold chosen on an inner validation split. Only `auc`, `ap` and `gmean_default` are analysed here; `gmean_moved` is published but not analysed, because it is a different decision rule |
| Script | **Not included.** The run cannot be reproduced from this repository. These are historical raw fold records only. The stored script is not shown to be the exact text that produced every dataset: the log shows two datasets first run with 40 folds and re-run with 100, and an ESR attempt abandoned before it finished (historical artifact, unverified here). The published file has 100 folds for every dataset |
| Data | Not included. The 13 datasets of the 15-dataset record come from the same loaders as section 3.1. The added datasets were loaded from third-party sources (see 8.2) |

### 8.2 Protocol, and how it differs from the 15-dataset record

Read from the run's script (protocol only):

| Element | Exploratory run | June four-way record |
|---|---|---|
| Datasets | 18: 13 of the 15 (ESR and DCCC, the two largest, were skipped by the script) plus mammography, thyroid_sick, coil_2000, satimage and creditcard | 15 |
| The 5 added datasets | External. mammography, thyroid_sick, coil_2000 and satimage were loaded through the imbalanced-learn dataset fetcher; creditcard from OpenML data id 1597, as a stratified subsample (all 492 positive rows plus 14,500 negative rows drawn with seed 42, n = 14,992). The script's comments call them benchmark datasets found by a search; no selection rule is recorded, so how these five were chosen is not established | none |
| Outer folds | RepeatedStratifiedKFold, 5 splits, 20 repeats, random_state 42: 100 folds | Same splitter and seed. That the outer test folds are identical for the 13 shared datasets was not checked |
| Inner split | Each outer training fold is split again, stratified 70 / 30 (random_state 42). **Every method is trained on the 70 percent part only** (56 percent of all rows). For CSRBoost, ERF and BRF the other 30 percent is used only to choose the threshold of the `gmean_moved` column: none of the three is fitted or configured on it, and the analysed columns `auc`, `ap` and `gmean_default` do not depend on it (read from the script) | No inner split. Methods are trained on the whole outer training fold (80 percent of all rows) |
| Scaler | StandardScaler fitted on the 70 percent part only | Fitted on the whole training fold |
| Threshold (analysed columns) | Fixed 0.5 for `gmean_default`; AUC and AP from probabilities | Fixed 0.5 |
| Seeds | CSRBoost 42; ERF 42 + 7919 b per bag; BRF 42; inner split 42 | The same rule for CSRBoost and ERF; RF-Raw and RF-BRF 42 |
| CSRBoost | The runner's functions (imported), AdaBoost of 50 stumps | Same code |
| ERF-CSRBoost | The runner's functions (imported): 20 bags of 50-tree random forests | Same code |
| BRF | BalancedRandomForest, **1,000 trees**, sqrt features, sampling_strategy "all", replacement True, bootstrap False, seed 42. It matches ERF's total tree count (20 x 50 = 1,000) but not its training cost: each BRF tree is grown on a balanced sample of 2 x n_min rows | RF-BRF has 100 trees |
| RF-Raw | Not in this run | Present |

Because of the inner split and the 1,000-tree BRF, numbers in this section are not comparable with
the June record's numbers, even for the same dataset. The two are never combined in this repository.

### 8.3 Results (exploratory, uncorrected)

Dataset means over 100 folds, then the unit of analysis is the dataset (N = 18). Wilcoxon signed-rank
tests are exact and two-sided (no zero differences, no tied absolute differences). The bootstrap
interval uses 10,000 resamples of the 18 dataset differences (NumPy `default_rng` seed 0). Six
tests are shown; **none is corrected for multiple comparisons**.

Global means (mean of the 18 dataset means):

| Metric (mean of the 18 dataset means) | CSRBoost | ERF-CSRBoost | BRF (1,000 trees) |
|---|---|---|---|
| AUC | 0.9204 | 0.9350 | 0.9380 |
| AP | 0.7060 | 0.7754 | 0.7744 |
| G-Mean at the fixed 0.5 threshold | 0.8517 | 0.8194 | 0.8839 |

ERF-CSRBoost against CSRBoost and against the tree-count-matched BRF:

| Comparison (unit = dataset, N = 18) | Mean difference | Median difference | Datasets higher / lower | Bootstrap 95% interval of the mean | Wilcoxon exact, two-sided p |
|---|---|---|---|---|---|
| AUC: ERF-CSRBoost minus CSRBoost | +0.0146 | +0.0118 | 15 / 3 | [+0.0062, +0.0230] | 0.0023 |
| AUC: ERF-CSRBoost minus BRF-1000 | -0.0030 | +0.0002 | 10 / 8 | [-0.0084, +0.0010] | 0.9323 |
| AP: ERF-CSRBoost minus CSRBoost | +0.0694 | +0.0598 | 16 / 2 | [+0.0361, +0.1062] | 0.0003 |
| AP: ERF-CSRBoost minus BRF-1000 | +0.0010 | +0.0026 | 12 / 6 | [-0.0171, +0.0180] | 0.3038 |
| G-Mean at 0.5: ERF-CSRBoost minus CSRBoost | -0.0323 | -0.0017 | 8 / 10 | [-0.0777, +0.0048] | 0.6705 |
| G-Mean at 0.5: ERF-CSRBoost minus BRF-1000 | -0.0645 | -0.0106 | 8 / 10 | [-0.1323, -0.0126] | 0.0304 |

Datasets where ERF-CSRBoost is lower than CSRBoost: on AUC BCW, FLAREF and coil_2000; on AP BCW
(-0.0012) and FLAREF (-0.0638). Datasets where ERF-CSRBoost is lower than BRF on AP: CARGOOD
(-0.1061), FLAREF (-0.0487), coil_2000 (-0.0240), CB, ILPD and, by less than 0.00005, WINE.

Per-dataset means (AUC, AP and G-Mean at 0.5; n, imbalance ratio and minority count as printed by
the run's own console log, historical artifact). "thesis record" in the Group column marks the 13
datasets that are also in section 3.1; "added" marks the five external datasets.

| Dataset | Group | n | Imbalance ratio | Minority | AUC CSRBoost | AUC ERF | AUC BRF | AP CSRBoost | AP ERF | AP BRF | G-Mean 0.5 CSRBoost | G-Mean 0.5 ERF | G-Mean 0.5 BRF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BCW | thesis record | 569 | 1.7 | 212 | 0.9904 | 0.9897 | 0.9898 | 0.9885 | 0.9873 | 0.9872 | 0.9581 | 0.9515 | 0.9505 |
| CARGOOD | thesis record | 1,728 | 24.0 | 69 | 0.9750 | 0.9889 | 0.9937 | 0.4858 | 0.7377 | 0.8438 | 0.9654 | 0.9327 | 0.9649 |
| CARVGOOD | thesis record | 1,728 | 25.6 | 65 | 0.9970 | 0.9998 | 0.9996 | 0.9224 | 0.9941 | 0.9911 | 0.9828 | 0.9953 | 0.9906 |
| CB | thesis record | 208 | 1.1 | 97 | 0.8636 | 0.9073 | 0.9098 | 0.8607 | 0.9050 | 0.9082 | 0.7772 | 0.8048 | 0.8009 |
| coil_2000 | added | 9,822 | 15.8 | 586 | 0.6990 | 0.6976 | 0.7363 | 0.1347 | 0.1391 | 0.1632 | 0.5456 | 0.3650 | 0.6238 |
| creditcard | added | 14,992 | 29.5 | 492 | 0.9732 | 0.9830 | 0.9790 | 0.8948 | 0.9189 | 0.9095 | 0.9328 | 0.9297 | 0.9349 |
| ESDRP | thesis record | 520 | 1.6 | 200 | 0.9702 | 0.9964 | 0.9962 | 0.9470 | 0.9948 | 0.9945 | 0.9069 | 0.9664 | 0.9654 |
| FLAREF | thesis record | 1,066 | 23.8 | 43 | 0.9012 | 0.8766 | 0.9000 | 0.3197 | 0.2559 | 0.3046 | 0.7765 | 0.5817 | 0.8226 |
| GLASS | thesis record | 196 | 5.8 | 29 | 0.9422 | 0.9746 | 0.9725 | 0.8681 | 0.9432 | 0.9363 | 0.8945 | 0.9212 | 0.9206 |
| ILPD | thesis record | 583 | 2.5 | 167 | 0.7207 | 0.7379 | 0.7368 | 0.4704 | 0.4938 | 0.4962 | 0.6609 | 0.6737 | 0.6685 |
| mammography | added | 11,183 | 42.0 | 260 | 0.9330 | 0.9472 | 0.9513 | 0.6158 | 0.7068 | 0.6803 | 0.8785 | 0.8681 | 0.8979 |
| PSDAS | thesis record | 4,424 | 4.6 | 794 | 0.7582 | 0.8134 | 0.8107 | 0.3985 | 0.4744 | 0.4689 | 0.6837 | 0.6834 | 0.7229 |
| satimage | added | 6,435 | 9.3 | 626 | 0.9226 | 0.9587 | 0.9532 | 0.5428 | 0.7612 | 0.7432 | 0.8386 | 0.8407 | 0.8720 |
| SEED | thesis record | 210 | 2.0 | 70 | 0.9753 | 0.9756 | 0.9746 | 0.9530 | 0.9567 | 0.9544 | 0.9178 | 0.9107 | 0.9099 |
| thyroid_sick | added | 3,772 | 15.3 | 231 | 0.9706 | 0.9931 | 0.9906 | 0.7784 | 0.9257 | 0.9029 | 0.9339 | 0.9352 | 0.9511 |
| WINE | thesis record | 178 | 2.7 | 48 | 0.9943 | 0.9997 | 0.9997 | 0.9863 | 0.9992 | 0.9992 | 0.9687 | 0.9842 | 0.9797 |
| YEAST5 | thesis record | 1,484 | 32.7 | 44 | 0.9824 | 0.9907 | 0.9914 | 0.6941 | 0.7935 | 0.7845 | 0.9291 | 0.9255 | 0.9561 |
| YEAST5ERL | thesis record | 1,484 | 295.8 | 5 | 0.9987 | 0.9998 | 0.9990 | 0.8468 | 0.9700 | 0.8717 | 0.7794 | 0.4800 | 0.9783 |

Gains concentrate in a few datasets. Leaving out the datasets with the largest gains:

| Datasets included | AP ERF minus CSRBoost | higher | p | AUC ERF minus CSRBoost | higher | p | AP ERF minus BRF | higher | p |
|---|---|---|---|---|---|---|---|---|---|
| All 18 datasets | +0.0694 | 16 / 18 | 0.0003 | +0.0146 | 15 / 18 | 0.0023 | +0.0010 | 12 / 18 | 0.3038 |
| Without CARGOOD | +0.0587 | 15 / 17 | 0.0007 | +0.0146 | 14 / 17 | 0.0038 | +0.0073 | 12 / 17 | 0.1202 |
| Without CARGOOD and satimage | +0.0487 | 14 / 16 | 0.0013 | +0.0133 | 13 / 16 | 0.0076 | +0.0066 | 11 / 16 | 0.1754 |
| Without CARGOOD, satimage and thyroid_sick | +0.0421 | 13 / 15 | 0.0026 | +0.0126 | 12 / 15 | 0.0125 | +0.0055 | 10 / 15 | 0.2524 |

By group (means of the dataset differences, no test run on a group):

| Group | Datasets | AP ERF minus CSRBoost | higher | AUC ERF minus CSRBoost | higher | AP ERF minus BRF | higher | AUC ERF minus BRF | higher |
|---|---|---|---|---|---|---|---|---|---|
| 13 datasets that are also in the thesis record | 13 | +0.0588 | 11 | +0.0139 | 11 | -0.0027 | 8 | -0.0018 | 7 |
| 5 added datasets | 5 | +0.0970 | 5 | +0.0162 | 4 | +0.0105 | 4 | -0.0062 | 3 |

### 8.4 What this run does and does not show

- **Against CSRBoost.** ERF-CSRBoost has a higher mean AP on 16 of 18 datasets (mean +0.0694, bootstrap
  interval +0.0361 to +0.1062, exact Wilcoxon p 0.0003, uncorrected) and a higher mean AUC on 15 of
  18. The gain shrinks but keeps its sign when CARGOOD, satimage and thyroid_sick are left out (AP
  +0.0421, 13 of 15). This comparison has **unequal capacity**: ERF-CSRBoost is 1,000 trees, CSRBoost
  is 50 stumps. It does not show that ERF-CSRBoost is a better method, only that this ensemble scores
  higher than this re-implemented booster.
- **Against the tree-count-matched balanced random forest.** There is no AUC or AP advantage: AUC
  -0.0030 (10 of 18 higher, p 0.93), AP +0.0010 (12 of 18 higher, p 0.30, interval -0.0171 to
  +0.0180). At the fixed 0.5 threshold ERF-CSRBoost has a lower G-Mean (-0.0645, 8 of 18 higher, p
  0.030, uncorrected, one of six tests shown). Leaving out CARGOOD, satimage and thyroid_sick gives an AP
  difference of +0.0055 over BRF (10 of 15 higher, p 0.25), which is not distinguishable from zero.
- **No other equal-capacity result exists.** The BRF comparison is the only one in which the
  competing model has ERF-CSRBoost's tree count. No comparison against a boosted model of equal
  capacity, and no ablation of ERF-CSRBoost's parts, was run.
- **Exploratory and uncorrected.** One run, six displayed tests with no multiplicity correction, p-values
  that treat the 18 datasets as independent (CARGOOD and CARVGOOD share features, as do YEAST5 and YEAST5ERL; sections 3.1 and 3.5) and a
  protocol that was fixed by the author. A small p-value here is a description of this run, not a
  confirmation.
- **Not independent for 13 of the 18 datasets.** The 13 datasets of the 15-dataset record were used
  throughout development (section 3.4). The five added datasets were chosen by the author and, as far
  as the author's records show, were not used in an earlier design step (unverified here); they were
  not drawn by a stated rule, and the run has no held-out test of its own.
- **Concentrated gains.** CARGOOD, satimage and thyroid_sick carry much of the AP gain over CSRBoost
  (49 percent of the sum of the 18 dataset differences). YEAST5ERL (5 positives, one per test fold) has
  the fourth-largest AP gain (+0.1232). Without these four datasets the mean AP difference is +0.0363
  over CSRBoost (12 of 14 higher, p 0.0052) and -0.0011 over BRF. FLAREF and BCW are lower. On the five added datasets the AP gain over CSRBoost is higher on 5 of 5 (a
  group of five cannot give a two-sided exact p below 0.0625).
- **Different protocol.** Models here are trained on 56 percent of the data, with a 1,000-tree BRF; the
  15-dataset record uses 80 percent and a 100-tree RF-BRF. The numbers cannot be combined or compared
  across the two.
- **No stored summary.** The original run left no summary file; all numbers above are recomputed from
  the raw rows.

## 9. Relationship to the audit

The audit repository (https://github.com/SherwanAli0/csrboost-audit) is a separate project that
studies the published CSRBoost method. Its figures (for example 42 of 143 metric-algorithm
combinations under a uniform protocol) concern that audit only. They are not ERF-CSRBoost results and
are not combined with the numbers above.

## 10. Recompute the main numbers

With the two files below in place and pandas and SciPy installed (this snippet was run against the
published files and printed the values quoted):

```python
import numpy as np
import pandas as pd
from scipy import stats

# June four-way record (sections 4.2 and 4.3)
d = pd.read_csv("research/results/comparison_june2026_four_way/fold_results.csv")
m = d.groupby(["dataset", "method"])[["acc", "auc", "f1", "ap", "gmean"]].mean()
print(m.groupby("method").mean().round(4))            # global means (mean of the 15 dataset means)

auc = m["auc"].unstack()                              # one row per dataset, one column per method
delta = auc["ERF_CSRBOOST"] - auc["CSRBOOST"]
print((delta > 0).sum(), round(delta.mean(), 4), round(stats.wilcoxon(delta).pvalue, 4))
print(round(stats.friedmanchisquare(*[auc[c] for c in auc.columns]).pvalue, 4))
rng = np.random.default_rng(0)                        # bootstrap interval of the mean difference (section 4.3)
idx = rng.integers(0, len(delta), size=(10000, len(delta)))
print(*np.percentile(delta.values[idx].mean(axis=1), [2.5, 97.5]).round(4))

# Exploratory 18-dataset run (section 8.3)
e = pd.read_csv("research/results/exploratory_18_dataset_run/fold_results.csv")
em = e.groupby(["dataset", "model"])[["auc", "ap", "gmean_default"]].mean()
for met in ["auc", "ap", "gmean_default"]:
    x = em[met].unstack()
    for other in ["CSRBoost", "BRF"]:
        dd = x["ERF"] - x[other]
        print(met, other, round(dd.mean(), 4), (dd > 0).sum(), round(stats.wilcoxon(dd, method="exact").pvalue, 4))
```

Expected output: the global means of section 4.2 (listed in alphabetical order of method), then
`12 0.0116 0.0302`, then `0.0054`, then `0.001 0.0221` (the bootstrap interval), then the six lines
`auc CSRBoost 0.0146 15 0.0023`, `auc BRF -0.003 10 0.9323`, `ap CSRBoost 0.0694 16 0.0003`, `ap BRF 0.001 12 0.3038`,
`gmean_default CSRBoost -0.0323 8 0.6705`, `gmean_default BRF -0.0645 8 0.0304`.

## 11. Release artifacts

SHA-256 of the stored bytes. The files were written on Windows with CRLF line endings and are kept
byte for byte by the repository's `.gitattributes` (`-text`). If a tool converts them to LF anyway,
the second column applies.

| File | SHA-256 of stored bytes | SHA-256 with LF endings |
|---|---|---|
| `research/results/comparison_june2026_four_way/fold_results.csv` (June four-way record) | `1918763389edc11b377e18dc1f91074431fbc1341706bba75fb06d0883213b44` | `875ff0eaa27089f289a8c3a774b4eafb4b2c0bb4155c3523821c4341121f9d65` |
| `research/results/comparison_june2026_four_way/dataset_summary.csv` | `9064a9d1d10c23cc3ff4f2ff84532070a2dce10a73c0c387b0df433591abc4fb` | `ff0633c039ed510f1c39ac4408f3da7d47f3fde2791650598403a9f9472f4f07` |
| `research/results/comparison_june2026_four_way/global_means.csv` | `507ca44fa4d9b9a87da0228656cb6bf7e75b70771a81d50a024ca50c06a93a48` | `8b238bc7c39bec169e16922e239904ca7c3e929f87f62ea41317a3a473fa5fd8` |
| `research/results/comparison_20260428_141315/fold_results.csv` (April run, earlier run) | `7d51467ddb7238bf1f38a8ca3883b884ee33902e34722578bee2120acaef4d8b` | `77a798662390f386df457196dc228ba82c8f92778ea778899d8191c3af8b5456` |
| `research/results/comparison_20260428_141315/dataset_summary.csv` | `76360bba3bd266e5d609804fc223524dc17b0d347db1c1d0bf7c3e12b43593d1` | `ce70824506302e98ee91f74cd696bdc9f6787a6fe6ee2960b5aa6cd6625ff428` |
| `research/results/exploratory_18_dataset_run/fold_results.csv` (derived: CSRBoost, ERF and BRF rows of the original file) | `560ef44c69abd862118237669f5d1c6a44a897ded571e46f4716d03bfe629a5f` | `ca0fa4e73dfa0a37be22b0852a5abec9882e2ef266d9ec87c2371e2aaab68407` |
| `research/scripts/comparison_rf_csrboost_erf.py` (selected runner) | `3a799cfa3fcc53e4e3cb6df8fc94fc8ebd53f4d409dad0ea246615ba838f42bd` | `e2eef829d96f62efdfeac7cb162ca5932bfd8c5cd0bbb25494b11f18f0e11fcf` |

Original file of the exploratory run (not included): 9,000 rows, SHA-256
`092857d64b228dd02f7fba697a7b7b7420cb6bbeed0e6e93a176004e80ddcb3f`. Thesis two-way file (not
included, equal to the CSRBoost and ERF-CSRBoost rows of the June record): SHA-256
`7b1e353bba72c4221fa5c811624111fe2d1b805683934e4c6475ce0ef4532ff2`.

The numbers in sections 2 to 8 were recomputed from the published files. The recomputation scripts
are not part of the repository; the snippet in section 10 reproduces the headline values.
