# Results evidence

What the stored ERF-CSRBoost results are, how they were produced, what was recomputed from the
saved records, and what cannot be concluded from them.

ERF-CSRBoost is an experimental extension of CSRBoost and an unpublished research prototype. The
numbers below are a historical artifact of an exploratory comparison on development datasets. They
are not an independent final evaluation and they do not establish a general improvement.

## 1. Status labels

| Label | Meaning |
|---|---|
| Historical artifact | A number or statement in a saved file that was read but not regenerated |
| Recomputed | Regenerated from the saved per-fold records (`fold_results.csv`) with Python, pandas and SciPy; no model was trained. The short script in section 10 reproduces the main numbers |
| Newly executed check | A new, cheap check run during release preparation on saved data files, with no model training |
| Read from code | A person read the runner script |
| Unresolved | Could not be established from the artifacts available |

## 2. The release record

| Item | Finding |
|---|---|
| Run identifier | `comparison_20260428_141315`. Started 28 April 2026 14:13:15 and finished about 29 April 2026 00:15; 36,102 s (10.0 hours) of wall clock according to the run log (historical artifact; the log is not part of the release set) |
| Methods | CSRBoost (re-implemented, run live in every fold), ERF-CSRBoost, RF-Raw, RF-BRF |
| Datasets | 15 |
| Records | `fold_results.csv`: 15 datasets x 4 methods x 100 folds = 6,000 rows, no missing values, no duplicate keys (recomputed) |
| Summary | `dataset_summary.csv`: per-dataset size, imbalance, and mean and standard deviation of each metric |
| Not recorded | Python and library versions of the run; per-method timing; any flag showing whether the SMOTE fallback ever triggered; the exact script text used for this run (see `IMPLEMENTATION_MAP.md`, section 3) |
| An aborted attempt | `comparison_20260428_141201` ran only one dataset for 40 rows and was superseded by the run above. It is not used |

Release artifacts and their SHA-256 are listed in section 11.

### A later corrected rerun (June 2026)

After the April run the author corrected three loader problems in the runner (ESR path, DCCC path
and the Sonar class polarity) and re-ran ESR, DCCC and CB. A two-method rerun (CSRBoost and
ERF-CSRBoost only, same 15 datasets, no RF baselines) was kept with the thesis submission. Twelve of
the 15 datasets are row-for-row identical between the April and June records (compared by a review
pass over the saved per-fold files). ESR and DCCC differ in
the fourth decimal. Only CB differs materially, because the positive class changed (see section 3).
Recomputed AUC means for the two shared methods (historical artifact, June records recomputed):

| Dataset | CSRBoost April | CSRBoost June | ERF April | ERF June |
|---|---|---|---|---|
| CB | 0.9028 | 0.9049 | 0.9362 | 0.9319 |
| ESR | 0.9776 | 0.9777 | 0.9960 | 0.9960 |
| DCCC | 0.7631 | 0.7630 | 0.7710 | 0.7710 |

In the June two-method records ERF-CSRBoost has a higher mean AUC than CSRBoost on 12 of 15
datasets (mean difference +0.0116, dataset-level Wilcoxon p 0.030; global mean AUC 0.9383 against
0.9267). That is the same picture as April. The June records are not the release record and have no
RF baselines, so they cannot say anything about RF-BRF.

## 3. Datasets and protocol

### 3.1 The 15 datasets

Counts come from the dataset files themselves (checked by a review pass) and agree with
`dataset_summary.csv` for n, features, minority and majority counts (recomputed for that file).
"Label 1" is the positive class used for every metric. The file hash column holds the first 12 hex
characters of the SHA-256 of the data file used, so a reader can check a downloaded copy.

| Key (run order) | Source of the file | n | Features | Label 1 means | Label 1 | Label 0 | Imbalance | File hash |
|---|---|---|---|---|---|---|---|---|
| PSDAS (1) | UCI | 4,424 | 36 | Enrolled (other two outcomes = 0) | 794 | 3,630 | 4.57 | `3ef126de5cef` |
| ESR (2) | UCI | 11,500 | 178 | class 1 (seizure) | 2,300 | 9,200 | 4.00 | `5ac55c71de5d` |
| DCCC (3) | UCI | 30,000 | 23 | default payment | 6,636 | 23,364 | 3.52 | `f61941714a35` |
| BCW (4) | UCI | 569 | 30 | M (malignant) | 212 | 357 | 1.68 | `d606af411f3e` |
| CB (5) | UCI | 208 | 60 | M (mine), the LARGER class | 111 | 97 | 1.14 | `e90434cdbf00` |
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

- **CB (Sonar).** In the run behind the stored table, label 1 is "M", the larger class (111 rows),
  so the resampling step treated the larger class as the minority and oversampled the smaller one
  (read from code, and from the class counts). AUC does not change when the positive class is swapped
  for the same scores, but AP, F1 and the balancing step do. The later corrected runner makes the
  smaller class (R, 97 rows) the positive class, which gives the June CB numbers in section 2.
- **ESDRP.** 269 of 520 rows are exact duplicates of other rows (features and class), counted from the
  file. A review pass found that, on average, 68.7 percent of test rows in the run's own folds have an
  exact twin in the training fold (newly executed check, not repeated here). Near-1.0 AUC for every
  method on ESDRP, and ERF's gain over CSRBoost there, should be read with that in mind.
- **FLARE-F.** 760 of 1,066 rows are exact duplicates of other rows (counted from the file). On average
  80.6 percent of test rows have an exact twin in the training fold (the same review pass). This is
  where ERF-CSRBoost loses to CSRBoost and RF-BRF. Duplicates plausibly affect every method, but the
  pattern is not explained.
- **YEAST5 and YEAST5ERL.** 31 duplicate rows each. YEAST5ERL has only 5 positives, so each stratified
  test fold holds exactly one positive; AUC, AP and G-Mean are then computed from a single positive,
  and F1 is 0 or 1 in every fold.
- **GLASS.** The file has 196 rows (types 1, 2, 5, 6 and 7 only). The original UCI glass data have 214
  rows. Whether this 196-row file is the glass dataset used by the CSRBoost paper is unresolved.
- **CARGOOD and CARVGOOD** share identical feature rows and differ only in which class is positive,
  so they are not independent datasets. The effective number of independent datasets is below 15.
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
forest) exists in this run. The comparison uses the same folds and the same metrics for all four
methods; it does not use the same compute budget.

### 3.3 Evaluation protocol (read from code, fold structure recomputed)

| Element | Finding |
|---|---|
| Split | RepeatedStratifiedKFold, 5 splits, 20 repeats, random_state 42: 100 train and test folds per dataset. All four methods see identical folds within a dataset (each method has fold indices 0 to 99 for every dataset) |
| Seeds | Splits 42. CSRBoost: 42 in every fold. ERF: 42 + 7919 b in every fold. RF-Raw and RF-BRF: 42 in every fold. The random components are therefore re-used identically across folds |
| Preprocessing | StandardScaler fitted on the training fold only; clustering and SMOTE fitted on the scaled training fold only; no resampling of test rows |
| SMOTE failure | An exception makes the runner silently use the retained rows without oversampling. The run does not record whether this ever happened |
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

## 4. Recomputed results

All numbers in this section are recomputed from `fold_results.csv`. Metrics are unweighted means over
100 folds per dataset. Fixed 0.5 threshold, untuned baselines, unequal capacity (section 3.2).

### 4.1 Mean AUC by dataset

| Dataset | CSRBoost | ERF-CSRBoost | RF-Raw | RF-BRF | ERF minus CSRBoost | ERF minus RF-BRF | ERF minus RF-Raw |
|---|---|---|---|---|---|---|---|
| PSDAS | 0.7601 | 0.8212 | 0.8081 | 0.8134 | +0.0611 | +0.0078 | +0.0131 |
| ESR | 0.9776 | 0.9960 | 0.9952 | 0.9946 | +0.0183 | +0.0014 | +0.0007 |
| DCCC | 0.7631 | 0.7710 | 0.7637 | 0.7727 | +0.0079 | -0.0017 | +0.0073 |
| BCW | 0.9922 | 0.9911 | 0.9897 | 0.9906 | -0.0011 | +0.0005 | +0.0014 |
| CB (larger class positive) | 0.9028 | 0.9362 | 0.9276 | 0.9243 | +0.0334 | +0.0119 | +0.0086 |
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
| CSRBoost (live) | 0.8951 | 0.9265 | 0.7262 | 0.7627 | 0.8701 |
| ERF-CSRBoost | 0.9185 | 0.9386 | 0.7579 | 0.8166 | 0.8432 |
| RF-Raw | 0.9259 | 0.9335 | 0.6776 | 0.8170 | 0.7252 |
| RF-BRF | 0.9009 | 0.9386 | 0.6971 | 0.8113 | 0.8910 |

At full precision ERF-CSRBoost is 0.938567 and RF-BRF is 0.938645, so RF-BRF is higher by 0.00008.
The two agree to four decimals. Accuracy, F1 and G-Mean at a fixed 0.5 threshold depend on how each
method calibrates probabilities and should not be read as ranking evidence.

### 4.3 Inference with the dataset as the unit (N = 15, AUC)

Folds of one dataset share training data and are positively correlated, so a test across all 1,500
pooled fold values overstates significance and is not used. The unit here is the dataset.

| Pair (unit = dataset, N = 15) | Mean difference | Datasets higher / lower | Wilcoxon signed-rank, two-sided p | Holm-adjusted p (family of three ERF comparisons) |
|---|---|---|---|---|
| ERF-CSRBoost minus CSRBoost | +0.0120 | 12 / 3 | 0.030 | 0.060 |
| ERF-CSRBoost minus RF-BRF | -0.0001 | 11 / 4 | 0.151 | 0.151 |
| ERF-CSRBoost minus RF-Raw | +0.0051 | 11 / 2 (2 ties) | 0.011 | 0.032 |

- A Friedman test over the four methods on the 15 dataset AUCs gives chi-square 12.69, p = 0.005.
  Average ranks (1 is best): ERF 1.67, RF-Raw 2.47, RF-BRF 2.53, CSRBoost 3.33.
- A bootstrap interval for the ERF minus CSRBoost mean AUC difference (10,000 resamples of the 15
  dataset differences) is roughly +0.002 to +0.023.
- The Holm family (three comparisons of ERF against each other method) was chosen after seeing the
  data. With all six pairwise comparisons the Holm-adjusted p for ERF against CSRBoost is about 0.12
  (computed by a review pass, not repeated here).
- Read every p-value in this section with three further limits. The models have unequal capacity
  (1,000 trees for ERF, 50 stumps for CSRBoost, 100 trees for RF-Raw and RF-BRF; no equal-capacity
  comparison was run). The 15 datasets were also used while the method was developed, so they are
  not a fresh test set. The unit is the dataset, with N = 15, and the datasets are not independent
  (CARGOOD and CARVGOOD share features; some have duplicate rows).
- The result is borderline and not robust to removing questionable datasets:

| Subset | ERF minus CSRBoost mean difference | Datasets higher | Wilcoxon p |
|---|---|---|---|
| All 15 | +0.0120 | 12 / 15 | 0.030 |
| Without YEAST5ERL (5 positives) | +0.0129 | 11 / 14 | 0.035 |
| Without CB (class polarity) | +0.0105 | 11 / 14 | 0.049 |
| Without ESDRP (duplicates) | +0.0112 | 11 / 14 | 0.049 |
| Without YEAST5ERL and CB | +0.0113 | 10 / 13 | 0.057 |

- CARGOOD and CARVGOOD share features, so N = 15 overstates independence.

### 4.4 Per-dataset paired results over the 100 folds

The corrected resampled t test of Nadeau and Bengio (variance factor 1/100 + 1/4, 99 degrees of
freedom) is used instead of a raw test over folds, with Holm adjustment across the 15 datasets. The
label rule is ours, chosen after seeing the data: "ERF better" or "ERF worse" only if the
Holm-adjusted corrected p is below 0.05 and the absolute mean difference is at least 0.005;
otherwise "tie or inconclusive".

ERF-CSRBoost minus CSRBoost, AUC:

| Dataset | Mean difference | Share of folds with ERF higher | Holm-adjusted corrected p | Label |
|---|---|---|---|---|
| PSDAS | +0.0611 | 1.00 | <0.0001 | ERF better |
| ESR | +0.0183 | 1.00 | <0.0001 | ERF better |
| DCCC | +0.0079 | 0.98 | 0.0098 | ERF better |
| BCW | -0.0011 | 0.44 | 1.0000 | tie or inconclusive |
| CB | +0.0334 | 0.80 | 0.7128 | tie or inconclusive |
| ESDRP | +0.0231 | 0.99 | 0.0044 | ERF better (duplicate-driven) |
| ILPD | +0.0261 | 0.76 | 0.7128 | tie or inconclusive |
| GLASS | +0.0245 | 0.56 | 1.0000 | tie or inconclusive |
| SEED | -0.0031 | 0.31 | 1.0000 | tie or inconclusive |
| WINE | +0.0007 | 0.10 | 1.0000 | tie or inconclusive |
| YEAST5ERL | +0.0004 | 0.11 | 1.0000 | tie or inconclusive |
| CARGOOD | +0.0172 | 1.00 | <0.0001 | ERF better |
| CARVGOOD | +0.0027 | 0.98 | 0.0503 | tie or inconclusive |
| YEAST5 | +0.0059 | 0.88 | 0.7128 | tie or inconclusive |
| FLAREF | -0.0368 | 0.11 | 0.3046 | tie or inconclusive (probable loss) |

Under the same rule, ERF against RF-BRF is "tie or inconclusive" on all 15 datasets, and ERF against
RF-Raw is "ERF better" on 2 datasets (PSDAS and DCCC) and inconclusive on the other 13 (recomputed).

### 4.5 Other metrics, ERF-CSRBoost minus CSRBoost (unit = dataset, not corrected across metrics)

| Metric | Mean difference | Datasets higher / lower | Wilcoxon p (uncorrected) |
|---|---|---|---|
| AP | +0.0539 | 12 / 3 | 0.0054 |
| F1 | +0.0316 | 10 / 5 | 0.0946 |
| G-Mean | -0.0269 | 9 / 6 | 0.8040 |
| ACC | +0.0235 | 12 / 3 | 0.0020 |

Two datasets drive the G-Mean averages: YEAST5ERL (one positive per test fold) and FLAREF. The AP
gain over CSRBoost is dominated by CARGOOD (CSRBoost AP 0.488, ERF 0.843; the dataset difference is
about +0.355); without CARGOOD the mean AP difference is about +0.032. There is no evidence that
ERF-CSRBoost helps G-Mean at the 0.5 threshold.

## 5. Where ERF-CSRBoost scores higher, ties and loses

Scope: 15 datasets, one protocol, fixed 0.5 threshold, untuned baselines, unequal capacity,
development datasets. Statements are about mean AUC unless a metric is named.

**Higher AUC than CSRBoost (not evidence of a better method).** Clear gains with fold-level agreement
on PSDAS (+0.061), ESR (+0.018), DCCC (+0.008) and CARGOOD (+0.017); ESDRP (+0.023) also passes the
rule but is duplicate-driven. Across datasets the mean AUC is higher on 12 of 15 and lower on 3 (BCW,
SEED and FLAREF; no ties) (mean +0.0120, Wilcoxon p 0.030, Holm 0.060 over a family chosen after
seeing the data and about 0.12 over all six pairs, Friedman p 0.005). The capacity, development-set
and dataset-level limits in section 4.3 apply. Against RF-Raw the mean AUC is +0.0051 (11 of the 13
non-tied datasets), which is partly a capacity difference (1,000 trees against 100).

**Ties.** Against RF-BRF there is no detectable difference (mean -0.0001, p 0.151). On saturated
datasets (every method at an AUC of about 0.97 or above: BCW, ESDRP, SEED, WINE, YEAST5ERL, CARVGOOD,
and GLASS at about 0.95 to 0.98) the methods cannot be separated, and any "win" there is a difference
in the third or fourth decimal. For that reason no count of per-dataset wins is given in this
document. ERF is higher by +0.006 to +0.033 on ILPD, CB, GLASS and YEAST5 against CSRBoost, but not
significantly under the corrected test.

**Loses or probably loses.** FLAREF: ERF is lower than CSRBoost by 0.0368 (higher in only 11 of 100
folds, corrected Holm p 0.30) and lower than RF-BRF by 0.0267; it is also the dataset with the most
duplicate rows. BCW (-0.0011) and SEED (-0.0031) favour CSRBoost by margins that are not meaningful.
DCCC, CARGOOD and YEAST5 against RF-BRF favour BRF by small margins. G-Mean at 0.5: ERF is lower than
CSRBoost (-0.0269) and RF-BRF (-0.0478) on the global mean, driven by YEAST5ERL and FLAREF; the
dataset-level tests are not significant.

## 6. Status of each claim made in this repository

| Claim | Status | Basis |
|---|---|---|
| The release record has 6,000 per-fold rows (15 x 4 x 100) with no missing values | Recomputed | `fold_results.csv` |
| Global means in section 4.2 | Recomputed | `fold_results.csv`. A `global_means.csv` file from the same run is not included in this repository |
| ERF-CSRBoost has a higher mean AUC than the re-implemented CSRBoost on 12 of 15 datasets | Recomputed | section 4.3 |
| Wilcoxon p 0.030, Holm 0.060, Friedman p 0.005 | Recomputed. The Holm family is a post-hoc choice, and the result is borderline | sections 4.3 |
| ERF-CSRBoost is not distinguishable from RF-BRF on AUC | Recomputed | section 4.3 |
| ERF-CSRBoost has a lower mean G-Mean than CSRBoost and RF-BRF at 0.5 | Recomputed (not significant at dataset level) | sections 4.2 and 4.5 |
| Five datasets show a clear ERF-over-CSRBoost gain | Recomputed, under a rule chosen after seeing the data | section 4.4 |
| The protocol is 5-fold stratified, 20 repeats, same folds for every method, scaler on the training fold, threshold 0.5 | Read from code; fold structure recomputed | section 3.3 |
| No threshold selection or tuning inside the run | Read from code | section 3.3 |
| ERF uses 1,000 trees against 50 stumps and 100 trees | Read from code | section 3.2 |
| In the run behind the table, CB's positive class is the larger class | Read from code and class counts | section 3.1 |
| ESDRP and FLARE-F contain 269 and 760 duplicate rows | Newly executed check (counted from the files) | section 3.1 |
| 68.7 percent (ESDRP) and 80.6 percent (FLARE-F) of test rows have a training twin | Newly executed check by a review pass; not repeated here | section 3.1 |
| The June rerun differs from April only materially on CB | Historical artifact, June records recomputed for the values shown | section 2 |
| The run took about 10 hours | Historical artifact (run log); not checkable from the release set | section 2 |
| Library versions of the run | Unresolved: not recorded | section 2 |
| The 15 datasets are those of the CSRBoost paper | Unresolved: not checked against the paper; GLASS differs in row count from the original | section 3.1 |
| Why ERF scores higher where it does (mechanism) | Unresolved and untested: no ablation separates the classifier change, bagging and the extra trees | not claimed |
| That ERF helps more at moderate imbalance, or RF-BRF at extreme imbalance | Unresolved and untested | not claimed |
| The application reproduces these results | Not the case. The app is adapted from the runner and uses a different protocol | `IMPLEMENTATION_MAP.md` |

## 7. Limitations to keep next to the tables

1. The datasets were reused throughout development. This is not an independent evaluation.
2. No equal-capacity baseline: 1,000 trees against 50 stumps and 100 trees. An equal-budget
   comparison is future work.
3. Fixed 0.5 threshold. Accuracy, F1 and G-Mean depend on calibration.
4. Duplicate rows (ESDRP, FLARE-F, and 31 in each YEAST5 file) can inflate test scores.
5. CB uses the larger class as positive in this run.
6. The GLASS file has 196 rows, not the 214 of the original UCI data.
7. CARGOOD and CARVGOOD share features. YEAST5ERL has 5 positives. The effective number of
   independent datasets is below 15.
8. The stored runner snapshot cannot load ESR and DCCC without path changes; the selected runner can.
9. Python and library versions of the April run were not recorded. Small datasets reproduce exactly
   across the April and June records; ESR and DCCC differ slightly between them, and the cause was not
   established. Bit-for-bit reproduction is not promised.
10. Per-method timing was not recorded.
11. The comparison method labelled CSRBoost is a local re-implementation (MiniBatchKMeans, depth-1
    stumps, no class-redefinition guard). It is not the authors' code and may not reproduce the
    paper's own numbers. The separate audit repository studies that question for the published method.

## 8. Other result sets that exist but are not part of this release

A two-method June rerun (section 2) and exploratory variants and experiments from later months
(other ensemble variants, extra datasets, a validation-split calibration) exist in the author's
working folders. They use different methods or protocols. They are not described here, they are not
plain ERF-CSRBoost, and no number from them is claimed.

## 9. Relationship to the audit

The audit repository (https://github.com/SherwanAli0/csrboost-audit) is a separate project that
studies the published CSRBoost method. Its figures (for example 42 of 143 metric-algorithm
combinations under a uniform protocol) concern that audit only. They are not ERF-CSRBoost results and
are not combined with the numbers above.

## 10. Recompute the main numbers

With `research/results/comparison_20260428_141315/fold_results.csv` in place and pandas and SciPy
installed (this snippet was run against the stored file and printed the values in sections 4.2 and 4.3):

```python
import pandas as pd
from scipy import stats

d = pd.read_csv("research/results/comparison_20260428_141315/fold_results.csv")
m = d.groupby(["dataset", "method"])[["acc", "auc", "f1", "ap", "gmean"]].mean()
print(m.groupby("method").mean().round(4))            # global means (mean of the 15 dataset means)

auc = m["auc"].unstack()                              # one row per dataset, one column per method
delta = auc["ERF_CSRBOOST"] - auc["CSRBOOST"]
print((delta > 0).sum(), round(delta.mean(), 4), round(stats.wilcoxon(delta).pvalue, 4))
print(round(stats.friedmanchisquare(*[auc[c] for c in auc.columns]).pvalue, 4))
```

Expected output: the global means of section 4.2, then `12 0.012 0.0302`, then `0.0054`.

## 11. Release artifacts

SHA-256 of the stored bytes. The files were written on Windows with CRLF line endings; a Git
checkout may convert them to LF, in which case the second column applies.

| File | SHA-256 of stored bytes | SHA-256 with LF endings |
|---|---|---|
| `research/results/comparison_20260428_141315/fold_results.csv` | `7d51467ddb7238bf1f38a8ca3883b884ee33902e34722578bee2120acaef4d8b` | `77a798662390f386df457196dc228ba82c8f92778ea778899d8191c3af8b5456` |
| `research/results/comparison_20260428_141315/dataset_summary.csv` | `76360bba3bd266e5d609804fc223524dc17b0d347db1c1d0bf7c3e12b43593d1` | `ce70824506302e98ee91f74cd696bdc9f6787a6fe6ee2960b5aa6cd6625ff428` |
| `research/results/june2026_two_method_rerun/fold_results.csv` (June two-method rerun, supporting record, not the release record) | `7b1e353bba72c4221fa5c811624111fe2d1b805683934e4c6475ce0ef4532ff2` | `717103c42e52331fb7a8db915791dc3eb2c3bb07f99289b01e05260ab9fa6407` |
| `research/scripts/comparison_rf_csrboost_erf.py` (selected runner) | `3a799cfa3fcc53e4e3cb6df8fc94fc8ebd53f4d409dad0ea246615ba838f42bd` | `e2eef829d96f62efdfeac7cb162ca5932bfd8c5cd0bbb25494b11f18f0e11fcf` |

The numbers in sections 4.1 to 4.5 were recomputed from `fold_results.csv`. The recomputation
scripts are not part of the repository; the snippet in section 10 reproduces the headline values.
