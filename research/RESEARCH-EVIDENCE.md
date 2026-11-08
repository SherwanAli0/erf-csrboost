# Research evidence: ERF-CSRBoost

Author: Sherwan Ali. Written 8 November 2026. Not peer reviewed.

This document holds the detail behind the short summaries in the README files: terms, evidence levels, the claims-and-evidence table, where the gain comes from, three further experiments, related work and corrections to earlier statements. Each caveat is stated once, in the place where it applies. The November record itself is [here](https://github.com/SherwanAli0/erf-csrboost/tree/main/research/results/november2026_aligned_extension); the older June record is described in `docs/RESULTS_EVIDENCE.md` of the repository.

## Terms

- Datasets. The 20 datasets are 18 distinct feature collections: CARGOOD and CARVGOOD share one feature table and differ only in which class is positive, and so do YEAST5 and YEAST5ERL, so they are not 20 independent samples. The file name `aligned_20_task_ap.csv` predates this wording and is kept unchanged.
- Reimplemented CSRBoost is this project's own code: MiniBatchKMeans clustered undersampling, SMOTE and 50 AdaBoost stumps. It is not the authors' code.
- A replication tries to reproduce a paper's published numbers; that is the separate [CSRBoost audit](https://github.com/SherwanAli0/csrboost-audit). The comparisons here put ERF against a reimplemented baseline and are not a replication.
- Counts are numbers of datasets. They are not significance tests, and no p-value is given for the 20 datasets.
- AP is average precision, taken as the mean over the 100 folds of each dataset. "ERF" is ERF-CSRBoost: 20 resampled bags, each a 50-tree random forest, with probabilities averaged and a fixed 0.5 threshold.

## Evidence levels

| Level | What it covers | Where |
|---|---|---|
| Public and checkable | The saved result tables (`fold_results.csv`, `aligned_20_task_ap.csv`, `protocol.json`, `completion.json`), expected fold coverage and the SHA-256 manifests. `verify_record.py` checks the saved tables and manifest hashes and runs no model | Public repository |
| Private | The prediction-level audit (all five metrics recomputed from every saved prediction file within 1e-12), the per-fit JSON records (they hold the Python versions), the adaptive-campaign records, the pilot cell records and the seven-dataset balanced-bagging run | Author's local workspace |
| Not publicly regenerable | The 2,000 November fits. They need `run.py`, `worker.py` and `legacy_adapter.py` (SHA-256 values in `protocol.json`) and the prepared dataset arrays, none of which is public. Only `legacy_source.py` is public, as `research/scripts/comparison_rf_csrboost_erf.py` | Cannot be rerun from the repository |

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| ERF has higher AP than reimplemented CSRBoost | Higher on 16 of 20 datasets, lower on 4 (BCW, SEED, FLAREF, coil_2000), no ties; mean difference +0.0685. Four datasets supply 65.7% of the summed difference. ERF has 1,000 trees, CSRBoost 50 stumps | Supported as a count. Not evidence of a better method (unequal capacity, concentrated gain) |
| ERF has higher AP than a 100-tree balanced random forest | Higher on 16, lower on 4 (CARGOOD, FLAREF, coil_2000, ILPD); mean difference +0.0093. 1,000 ERF trees against 100 | Count supported. Capacity unequal |
| ERF has higher AP than a 100-tree plain random forest | Higher on 12, tied on 2 (WINE, YEAST5ERL), lower on 6 (CARGOOD, ESDRP, YEAST5, mammography, satimage, thyroid_sick). Mean AP 0.7872 for ERF against 0.7880 for the plain forest, a mean AP difference of -0.0008 | Not supported. The mean AP difference is slightly negative; the result shows neither an advantage nor equivalence |
| ERF has higher AP than a tree-count-matched balanced random forest (1,000 trees) | Experiment (a) below: 18 datasets, mean AP difference +0.0010, bootstrap 95% interval -0.0171 to +0.0180, higher on 12 and lower on 6; AUC and G-Mean at 0.5 lower for ERF | No advantage shown (the interval includes zero). Tree count matched, run time neither matched nor recorded; different protocol; its script is not published |
| Each ERF component (clustering, SMOTE, bagging) is needed | Experiment (c) below, a 144-cell pilot (separate implementation and protocol): ERF without clustering scored higher mean AP than ERF on 4 of 5 dataset families (tied on WINE); without SMOTE it was higher on 2, lower on 2 and tied on 1; with one bag it was lower on 4 and tied on 1 | Clustering: not supported in the pilot. Bagging: supported in the pilot. SMOTE: mixed in the pilot. All small and exploratory; untested on the 20 datasets |
| ERF costs more to run | Median paired fit-and-score time is 6.6 to 35.6 times that of a plain random forest and 15.0 to 25.1 times that of CSRBoost, depending on the dataset (five new datasets only, one thread; each ratio pairs the methods on the same folds within one dataset's run; two environments were used, see Cost and environment below) | Supported as observed wall-clock time. Not a controlled benchmark |
| Adaptive sampling or adaptive ensemble size improves ERF | Separate 480-cell campaign: 382 newly successful, 96 reused and 2 failed (both YEAST5ERL). Against fixed ERF on recall at 10% review, on two-seed stratified dataset means: adaptive sampling higher on 6, tied on 12, lower on 2; adaptive size 3 / 12 / 5; both 5 / 10 / 5. Exploratory, different protocol | Private records; not checkable publicly |
| The 2,000 November fits are complete | 2,000 rows, all with status success, 100 folds for each of 20 dataset-method cells, no recorded SMOTE fallback. A private audit recomputed all five metrics from every saved prediction file within 1e-12 | Completeness is publicly checkable. Prediction-level correctness is private. The fits cannot be rerun from the repository |
| CSRBoost audit: 42 of 143 published result rows reproduce within 3 points under one uniform protocol | Separate project. 42 of 143 rows (29.4%); the other 101 rows of its final table are tuned values that were not regenerated | Supported for the audit only. Says nothing about ERF |

## Where the gain comes from

The mean AP difference between ERF and CSRBoost over the 20 datasets is +0.0685 (sum +1.3705); the largest per-dataset gain is +0.3548 (CARGOOD) and the largest loss is -0.0579 (FLAREF). Four datasets supply 65.7% of that sum:

| Dataset | ERF minus CSRBoost (AP) | Share of the summed difference | ERF minus plain random forest (AP) |
|---|---:|---:|---:|
| CARGOOD | +0.3548 | 25.9% | -0.0543 |
| satimage | +0.2518 | 18.4% | -0.0024 |
| thyroid_sick | +0.1633 | 11.9% | -0.0117 |
| mammography | +0.1301 | 9.5% | -0.0135 |

On the other 16 datasets the mean difference over CSRBoost is +0.0294 (higher on 12, lower on 4) and the mean difference from a plain random forest is +0.0041. On the four datasets above a plain 100-tree random forest scores higher than ERF. Over all 20 datasets a plain random forest gains +0.0693 over CSRBoost (higher on 16, lower on 4), slightly more than ERF's +0.0685. The four datasets were picked out after seeing the results, so the 65.7% share is descriptive. These facts are consistent with the gain coming from the random forest classifier rather than from the resampling or the bagging, but that is not established on the 20 datasets; experiments (a) to (c) below are the only component checks.

## Three further experiments

None of them changes the 20-dataset table. They are listed with their limits so that they can be weighed, not as headline results.

### (a) Older 18-dataset run against a 1,000-tree balanced random forest

- What it is: 13 of the 15 June datasets (ESR and DCCC were skipped) plus mammography, thyroid_sick, coil_2000, satimage and creditcard, with CSRBoost, ERF and a 1,000-tree balanced random forest (BRF-1000), 100 folds each. The published file is `research/results/exploratory_18_dataset_run/fold_results.csv` (5,400 rows). BRF-1000 matches ERF's total tree count but not its training cost, and the file has no timing column (columns `dataset, src, fold, model, auc, ap, gmean_default, gmean_moved, pick`), so run time was not compared.
- Result: mean AP is ERF 0.7754, BRF-1000 0.7744 and CSRBoost 0.7060. ERF minus BRF-1000 is +0.0010, with ERF higher on 12 datasets and lower on 6 (exact Wilcoxon p = 0.3038; bootstrap 95% interval -0.0171 to +0.0180). Mean AUC difference is -0.0030 (10 higher, 8 lower). G-Mean at the 0.5 threshold is -0.0645 (p = 0.0304, uncorrected).
- Limits: each method is trained on 70% of each outer training fold, so the numbers are not comparable with the June record; the script that produced it is not published; no selection rule is recorded for the five added datasets; the test results are uncorrected for multiple comparisons.
- Reading: on AP the data do not separate ERF from a tree-count-matched balanced forest, and ERF is lower on AUC and on G-Mean at 0.5. This is a tree-count-matched comparison, not a matched-compute one.
- Evidence level: public raw records; the script is not public.

### (b) Seven-dataset check with a balanced-bagging variant (incomplete; not a result)

- What it is: ERF, BRF-1000, BRF-100 and BBAG_RF on GLASS, SEED, WINE, YEAST5ERL, YEAST5, FLAREF and CARVGOOD, 20 folds (5 x 4) each, 560 rows. The file is in the author's local graduation-project folder and is not in this repository. BBAG_RF is imbalanced-learn's `BalancedBaggingClassifier` with 20 bags of 50-tree forests (`fair_comparison.py`, lines 64-72). With the library defaults, each bag is a bootstrap sample of the training rows that is then randomly undersampled so the majority class matches the minority class count of that sample, with no clustering and no SMOTE. ERF instead keeps about half of every majority cluster and adds SMOTE minority points to the retained majority count (`novelty final\scripts\comparison_rf_csrboost_erf.py`, lines 457-474 and 583-602). BBAG_RF is therefore a different sampling scheme, not ERF with parts removed.
- Descriptive result (not a test): on the six datasets with complete BBAG_RF data (all except YEAST5ERL), the mean AP difference of ERF minus BBAG_RF is +0.0014 (ERF 0.8259, BBAG_RF 0.8245; ERF higher on 3 datasets, lower on 2, tied on WINE). Per dataset: YEAST5 +0.0482, CARVGOOD +0.0059, SEED +0.0032, WINE 0.0000, GLASS -0.0017, FLAREF -0.0472. On the same six datasets the mean AP of BRF-1000 is 0.8279 and of BRF-100 0.8207. No p-value is given. Over all seven datasets (complete for ERF and BRF-1000), mean AUC is 0.9735 for ERF against 0.9778 for BRF-1000 and G-Mean at 0.5 is 0.8465 against 0.9406.
- Limits: BBAG_RF produced no metrics in 6 of the 20 YEAST5ERL folds (every metric missing; the other 14 folds are valid), so that dataset is left out of the comparison above. The runner `run_ablation_20fold.py` (docstring, lines 1-2, and the ORDER filter, line 20) planned 13 datasets x 20 folds (all except ESR and DCCC). The run was interrupted in YEAST5ERL after GLASS, SEED and WINE had finished (`run20.log` shows two starts), and `run_ablation_hard.py` then completed the four hard datasets, so 7 datasets x 20 folds exist and six planned datasets (CB, ILPD, ESDRP, BCW, CARGOOD, PSDAS) were never run. The 100-fold default of `fair_comparison.py` (15 datasets) was not used. This run is incomplete and is not cited as evidence for or against ERF.
- Evidence level: private.

### (c) Controlled 144-cell pilot with component ablations (separate implementation and protocol)

- What it is: 12 model configurations on five dataset families (BCW, SONAR, GLASS, WINE, YEAST5) with two stratified splits each (seeds 42 and 142) plus two feature-group splits of YEAST5: 144 cells, all completed. The summary is `research/PILOT-RESULTS.md` in the separate local ERF workspace (not in this repository).
- Implementation and protocol: the pilot uses the newer `rare_event_eval` package (version 0.2.0) and its own `ERFClassifier` (`estimators.py`, line 74), not the runner behind the June and November records. The primary stratified splits use approximately 60/20/20 into train, validation and test (`splits.py`, lines 40-41; `pilot.py`, line 34); the pilot also includes feature-group splits (two seeds of YEAST5). On the primary splits, models are fitted on the 60% train part, the validation part only selects a threshold, and average precision is computed on the 20% test part (`training.py`, lines 132-145). The historical comparisons use 5 x 20 repeated stratified cross-validation (100 folds) in which each method trains on 80% of the rows (`docs/RESULTS_EVIDENCE.md`, sections 3.3 and 8.2). Pilot numbers and historical numbers are not comparable and are not combined here.
- Result (mean AP over the five families, stratified splits): ERF 0.9400; ERF without clustering 0.9488; ERF without SMOTE 0.9440; ERF with one bag 0.9261; plain random forest 0.9465; 1,000-tree weighted random forest 0.9421; balanced random forest 0.9183. Per family against ERF: without clustering higher on 4 and tied on WINE; without SMOTE higher on 2, lower on 2 and tied on WINE; one bag lower on 4 and tied on WINE; plain random forest higher on 3 (GLASS, SONAR, YEAST5), lower on 1 (BCW) and tied on WINE; 1,000-tree weighted forest higher on 2 (SONAR, GLASS), lower on 2 (BCW, YEAST5) and tied on WINE.
- Reading: in this pilot the clustering step did not help. Removing it did not lower mean AP on any of the five families, and removing SMOTE did not lower it consistently. Reducing the ensemble to one bag lowered mean AP on 4 of 5 families, so bagging was supported in the pilot. The plain random forest also scored above ERF on 3 of 5 families. These results come from a different implementation and partition scheme and do not confirm or explain the 20-dataset results. No default was changed; testing the clustering step on the 20 datasets is an open next step.
- Confirmation gate: ERF had higher mean AP than the balanced random forest on 4 of 5 families (BCW +0.0030, SONAR +0.0115, GLASS +0.0049, YEAST5 +0.0891; WINE tied). The predeclared rule for a larger confirmation study required ERF to beat the balanced random forest by at least 0.01 AP on four families and a median paired fit-time ratio (ERF over balanced random forest) of at most 3. The AP condition was met on 2 of 5 families (SONAR and YEAST5) and the observed median ratio over the ten primary runs was 5.97, so the gate was not met.
- Limits: two correlated seeds per family, five families that had already been inspected and small positive counts.
- Evidence level: private cell records; aggregates only in the author's workspace.

## Cost, environment and the adaptive campaign

- Cost: the median paired ERF / plain random forest fit-and-score time is 25.31x (coil_2000), 6.60x (creditcard), 14.91x (mammography), 35.59x (satimage) and 17.36x (thyroid_sick). Each ratio pairs the methods on the same folds within one dataset's run, so it does not compare machines. The public June record has no per-fit timing, so cost is not measured for the 15 June datasets.
- Environment: the Python and library versions of the April and June runs were not recorded. The November runs recorded library versions in `protocol.json` (numpy 2.2.6, scipy 1.18.1, pandas 2.3.3, scikit-learn 1.8.0, imbalanced-learn 0.14.1, joblib 1.6.0, threadpoolctl 3.8.0). The private per-fit files record Python and platform for all 2,000 fits: Python 3.12.14 on Windows-11-10.0.26200-SP0 for the 800 mammography and thyroid_sick fits (execution lane `laptop` on 667 of them; the lane field is missing on 133 mammography records, which carry the same platform string), and Python 3.13.18 on Linux-6.6.159+ for the 1,200 coil_2000, creditcard and satimage fits (lanes colab-coil, colab-creditcard and colab-satimage, Google Colab). Two environments were therefore used, not one per dataset. The November runs used one thread (n_jobs=1); the historical runner used n_jobs=-1.
- Adaptive campaign: 480 cells planned: 382 newly successful, 96 reused and 2 failed. Both failures are YEAST5ERL feature-group cells (seed 142, procedures AF and AA) where the inner split lacks both classes and no outer-validation fallback was used. The counts come from a private summary file.

## Related work

Combining resampling with forests or ensembles is established, so ERF's originality does not rest on that combination. Balanced Random Forest (Chen, Liaw and Breiman, 2004, UC Berkeley Technical Report 666) trains each tree on a class-balanced sample and is one of the baselines here. RUSBoost (Seiffert, Khoshgoftaar, Van Hulse and Napolitano, IEEE Transactions on Systems, Man, and Cybernetics, Part A, 40(1):185-197, 2010, doi 10.1109/TSMCA.2009.2029559) combines random undersampling with boosting. iBRF (Newaz, Mohosheu, al Noman and Jabid, arXiv:2403.09867, 2024, cited as an arXiv preprint) combines neighbourhood cleaning, random undersampling and SMOTE inside a balanced random forest. CSBBoost (Salehi and Khedmati, Scientific Reports 14:5152, 2024, doi 10.1038/s41598-024-55598-1) applies k-means to each class, undersamples the majority clusters proportionally, applies SMOTE to each minority cluster and then trains random forest and XGBoost models. ERF is built on CSRBoost (Yadav et al., IEEE Access, 2025): it keeps a CSRBoost-style balancing step and replaces AdaBoost with 20 resampled 50-tree forests whose probabilities are averaged. None of iBRF, CSBBoost or RUSBoost was run against ERF here. What this work contributes is the exact ERF construction and its documented evaluation, including where it does not help. It does not show that ERF improves on these methods or on a plain random forest. The sources above are the ones checked; other related methods may exist.

## Corrections to earlier statements

| Earlier statement | Where it appeared | Now |
|---|---|---|
| "17 of 20 datasets, AP +0.064" | Earlier ERF text | Retired: it combined two protocols. The current comparison is 16 of 20 on one aligned footing |
| "2,000 verified fits" | Profile README | "Completed". The public files verify the saved tables; prediction-level verification is private |
| "Mixed raw-RF performance" | Result-record banners | Replaced by the counts: higher on 12, tied on 2, lower on 6, mean AP difference -0.0008 |
| "Python 3.12 laptop and cloud execution" | November record README | Python 3.12.14 (800 fits) and 3.13.18 (1,200 fits) |
| "Library versions of the stored runs were not recorded" | Main README and research README | True for the April and June runs only; the November runs recorded library versions |
| "The June four-way record is the headline" | Main README, research README, docs | Called the historical AUC record; the latest comparison is the November AP extension |
| "The app now integrates the same Python evaluation core" | Workspace FINAL-REVIEW.md | The package sits beside the unchanged legacy app; only the offline workbench uses it |
| "18 bookkeeping feature collections" | November record README | "18 distinct feature collections" |
