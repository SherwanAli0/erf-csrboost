# November 2026 ERF evaluation record

Author: Sherwan Ali. Updated 7 November 2026; wording and disclosures revised 8 November 2026 (no result changed).

The five-dataset extension completed 2,000 new fits: five datasets x four methods x 100 folds. All succeeded, with zero errors, timeouts or recorded SMOTE fallbacks. These results are combined descriptively with the original 15-dataset June record. The work is not peer reviewed, and its limits are stated once, under Scope and limits below. Claims, experiments, related work and corrections are in [RESEARCH-EVIDENCE.md](../../RESEARCH-EVIDENCE.md). The public files let you check the saved tables, not rerun the fits.

## Highlights

- ERF exceeds the reimplemented CSRBoost on 16 of 20 datasets (+0.0685 mean AP; the largest per-dataset gain is +0.3548 on CARGOOD and the largest loss is -0.0579 on FLAREF) and a 100-tree balanced random forest on 16 of 20 datasets (+0.0093 mean AP).
- ERF is within 0.001 mean AP of a 100-tree plain random forest (0.7872 against 0.7880; ERF higher on 12 datasets, tied on 2, lower on 6), and it has the highest mean AP of the four methods on 7 datasets and is tied for highest on 2 more.
- All 2,000 new fits (five datasets, four methods, 100 folds each) completed with zero errors, timeouts or recorded SMOTE fallbacks; all five metrics of every fit were recomputed from the saved predictions within a tolerance of 1e-12 (done on private prediction files, see Scope and limits); and every table of this record is covered by `MANIFEST.sha256` and the read-only checker `verify_record.py`.
- ERF's own controls are reported next to the headline. In a tree-count-matched run against a 1,000-tree balanced random forest (a different protocol: each method trains on 70% of each outer training fold) the mean AP difference was +0.0010 over 18 datasets (ERF higher on 12, lower on 6; bootstrap 95% interval -0.0171 to +0.0180).
- In a separate 144-cell controlled pilot (approximately 60/20/20 train, validation and test partitions and newer evaluation code, so a different protocol again) ERF had higher mean AP than a balanced random forest on 4 of 5 dataset families (the fifth, WINE, tied). The pilot's predeclared confirmation gate required at least 0.01 AP over the balanced forest on 4 families and a median paired ERF-to-balanced-forest fit-time ratio of at most 3; ERF met the AP condition on 2 of 5 families and the observed time ratio was 5.97, so the gate was not met. A plain random forest had higher mean AP than ERF on 3 of 5 families (SONAR, GLASS and YEAST5), and most of ERF's advantage over the balanced forest comes from YEAST5 (+0.0891, against an average of +0.0049 on the other four families). The pilot's cell records are private; both controls and the pilot aggregates are described in [RESEARCH-EVIDENCE.md](../../RESEARCH-EVIDENCE.md).

## Main result

Mean average precision (AP) is compared after averaging the 100 folds within each dataset. Win/tie/loss counts are descriptive; differences within 1e-12 are treated as numerical ties.

| Baseline | ERF higher AP | Numerical ties | ERF lower AP | Mean dataset AP difference |
|---|---:|---:|---:|---:|
| CSRBOOST | 16 | 0 | 4 | +0.068523 |
| RF_BRF | 16 | 0 | 4 | +0.009251 |
| RF_RAW | 12 | 2 | 6 | -0.000777 |

- ERF has the strictly highest mean AP of the four methods on seven datasets and is tied for highest on two more. Against plain Random Forest the mean AP difference is -0.0008 (ERF higher on 12 datasets, tied on 2 [WINE and YEAST5ERL, both 1.000000], lower on 6).
- Median fit-and-score time is 6.6 to 35.6 times that of plain Random Forest (cost table below). How the gain over CSRBoost is distributed, and how it compares with the gain of a plain Random Forest, is stated once under Scope and limits.

The updated comparison is 16/20 higher-AP datasets versus the local CSRBoost reimplementation: 12/15 historical datasets plus 4/5 newly aligned datasets. The old 17/20 statement combined different protocols and remains retired. The 18-dataset July record remains a separate exploratory record. The small coil loss (-0.0001196 AP) is a descriptive numerical loss, not established meaningful inferiority.

## Scope and limits

- Not peer reviewed. The counts are descriptive, not significance tests, and the near-tie with a plain random forest shows neither an advantage nor equivalence (it does not establish an overall advantage for ERF). The 20 datasets were also used while the method was developed, so they are not an independent test, and they are 18 distinct feature collections: CARGOOD and CARVGOOD share one feature table and differ only in which class is positive, and so do YEAST5 and YEAST5ERL, so they are not 20 independent samples. Duplicate and train/test feature twins, conflicting labels, rare positives and the 196-row Glass variant remain as data caveats.
- Model capacity is unequal (ERF 1,000 trees, the forests 100 trees, CSRBoost 50 stumps), which prevents attributing gains to the sampling mechanism alone. A tree-count-matched comparison exists (see the controls bullet under Highlights) and no matched-compute study has been completed; a matched-compute comparison and component ablations on all 20 datasets are the next research direction. The smaller checks (an incomplete seven-dataset balanced-bagging run and the 144-cell pilot) are in [RESEARCH-EVIDENCE.md](../../RESEARCH-EVIDENCE.md).
- The gain over CSRBoost is concentrated in four datasets (CARGOOD, satimage, thyroid_sick, mammography: 65.7% of the summed gain), a plain Random Forest gains slightly more over CSRBoost (+0.0693) than ERF does (+0.0685), and the cause is untested.
- Software environments differ: this is not a fresh rerun of all 20 datasets in one environment. The June runs used an unrecorded environment, and the 2,000 new fits ran on Python 3.12.14 (800 fits, mammography and thyroid_sick, on a Windows 11 laptop) and 3.13.18 (1,200 fits, 400 each for coil_2000, satimage and creditcard, on Google Colab on Linux), as recorded in the private per-fit files; Kaggle was not used for this extension, and the public `protocol.json` records library versions only.
- Dataset arrays, row-level predictions, per-fit result files, the prediction-level verification and the code that ran the 2,000 fits (`run.py`, `worker.py`, `legacy_adapter.py`) are private, so those fits cannot be rerun from the public repository.

## All dataset means

| Dataset | CSRBoost | ERF | Raw RF | Balanced RF | Source |
|---|---:|---:|---:|---:|---|
| BCW | 0.990088 | 0.989491 | 0.987517 | 0.988441 | historical_june_15 |
| CARGOOD | 0.488315 | 0.843099 | 0.897355 | 0.882231 | historical_june_15 |
| CARVGOOD | 0.927977 | 0.998129 | 0.997646 | 0.994467 | historical_june_15 |
| CB | 0.901285 | 0.927784 | 0.922558 | 0.922539 | historical_june_15 |
| DCCC | 0.530813 | 0.540854 | 0.535161 | 0.539835 | historical_june_15 |
| ESDRP | 0.957295 | 0.997744 | 0.997775 | 0.997446 | historical_june_15 |
| ESR | 0.925640 | 0.985423 | 0.983234 | 0.978998 | historical_june_15 |
| FLAREF | 0.304204 | 0.246352 | 0.231150 | 0.287720 | historical_june_15 |
| GLASS | 0.899337 | 0.945806 | 0.942560 | 0.942618 | historical_june_15 |
| ILPD | 0.469411 | 0.500863 | 0.492472 | 0.503691 | historical_june_15 |
| PSDAS | 0.396759 | 0.487880 | 0.483638 | 0.466374 | historical_june_15 |
| SEED | 0.970282 | 0.964556 | 0.962537 | 0.959011 | historical_june_15 |
| WINE | 0.998097 | 1.000000 | 1.000000 | 0.999540 | historical_june_15 |
| YEAST5 | 0.730352 | 0.799215 | 0.803857 | 0.770590 | historical_june_15 |
| YEAST5ERL | 0.937500 | 1.000000 | 1.000000 | 0.920595 | historical_june_15 |
| coil_2000 | 0.140971 | 0.140851 | 0.135324 | 0.161056 | new_june_protocol_extension_5 |
| creditcard | 0.897751 | 0.923345 | 0.906488 | 0.911826 | new_june_protocol_extension_5 |
| mammography | 0.603837 | 0.733907 | 0.747400 | 0.679633 | new_june_protocol_extension_5 |
| satimage | 0.527681 | 0.779480 | 0.781911 | 0.746167 | new_june_protocol_extension_5 |
| thyroid_sick | 0.776021 | 0.939300 | 0.951035 | 0.906277 | new_june_protocol_extension_5 |

## Newly executed five-dataset protocol

- RepeatedStratifiedKFold: five folds, 20 repeats, seed 42; 80% training and 20% test.
- StandardScaler is fitted on each training fold only. Fixed 0.5 classification threshold; no inner validation, adaptive selection, calibration or threshold optimization.
- CSRBoost: MiniBatchKMeans with k equal to minority count, n_init=1, max_iter=100, batch_size=1024; retain ceil(50%) per cluster, SMOTE, then 50 AdaBoost stumps.
- ERF: 20 independently resampled bags, 50 RF trees per bag; bag seeds 42 + 7919*b. Raw RF and Balanced RF each use 100 trees (capacity: see Scope and limits).
- One computational thread. Historical resampling fallback behavior is preserved but instrumented; no fallback occurred in the 2,000 new fits.
- Metrics: accuracy, ROC AUC, F1, AP and G-mean. Full new fold metrics, including failed-cell status fields, are in fold_results.csv; there are no failed cells in this completed record.
- Pinned dependencies (protocol.json): numpy 2.2.6, scipy 1.18.1, pandas 2.3.3, scikit-learn 1.8.0, imbalanced-learn 0.14.1, joblib 1.6.0, threadpoolctl 3.8.0. Python versions and environments are described under Scope and limits. The historical runner used n_jobs=-1; these runs used one thread.

| Added dataset | Rows | Features | Positive rows |
|---|---:|---:|---:|
| mammography | 11183 | 6 | 260 |
| thyroid_sick | 3772 | 52 | 231 |
| coil_2000 | 9822 | 85 | 586 |
| satimage | 6435 | 36 | 626 |
| creditcard | 14992 | 30 | 492 |

Creditcard is the historical 14,992-row subset: all 492 fraud rows plus 14,500 nonfraud rows selected without replacement with RandomState(42), then concatenated/shuffled using the preserved preparation. It is not the full credit-card source dataset. Prepared artifact and split hashes are recorded in protocol.json. The orchestration code (run.py, worker.py, legacy_adapter.py) is not in this repository (see Scope and limits); protocol.json records its SHA-256 values, and its legacy_source.py entry is the public runner research/scripts/comparison_rf_csrboost_erf.py in the erf-csrboost repository (SHA-256 3a799cfa3fcc53e4e3cb6df8fc94fc8ebd53f4d409dad0ea246615ba838f42bd).

## Cost and interpretation

| Added dataset | Median paired ERF / raw-RF fit-and-score time |
|---|---:|
| coil_2000 | 25.31x |
| creditcard | 6.60x |
| mammography | 14.91x |
| satimage | 35.59x |
| thyroid_sick | 17.36x |

Timing ratios pair the same folds within each dataset's run. They are observed wall-clock fit-and-score costs, not controlled compute-budget benchmarks or cross-hardware performance claims. Over the same five datasets the median paired ERF / CSRBoost time ratio is 15.0x to 25.1x (fold_results.csv, column fit_and_score_seconds). The public June record has no per-fit timing, so cost is not measured for the 15 June datasets. The next research direction and the smaller existing checks are listed under Scope and limits.

The original data caveats remain (see Scope and limits). Historical June AUC findings are still valid as that separate record; they must not be relabelled as the new AP result. No pooled-fold p-value, claim of general advantage or publication claim is made. The app has its own preprocessing and evaluation and was not changed or redeployed for this release.

## Separate adaptive campaign

A separate 480-cell campaign evaluated fixed ERF, adaptive sampling, adaptive ensemble size and their combination. It completed with 382 newly executed successes, 96 reused results and two explicit errors; zero pending/timeouts. The two failures are YEAST5ERL feature-group cells (seed 142, procedures AF and AA) in which the inner split lacks both classes and no outer-validation fallback was used. Its primary recall-at-10%-review comparison is against fixed ERF, not CSRBoost: on stratified two-seed dataset means, adaptive sampling gave 6 higher / 12 tied / 2 lower datasets, adaptive size 3 / 12 / 5, and both 5 / 10 / 5. This is exploratory, uses a different protocol and is not pooled with the 2,000-fit AP extension. Full adaptive source/results remain a separate local record; this supplement does not claim to release or independently regenerate that campaign. Its counts (480 planned: 382 newly successful, 96 reused, 2 failed) come from a private summary file and cannot be checked from the public files.

## Verification and scope

The local completion audit checked all 2,000 job identities, frozen source/data/split hashes, ZIP CRCs, prediction hashes, exact test indices/labels, population sizes and recomputed all five metrics from every saved prediction artifact within 1e-12 tolerance. Original final archives remain preserved locally. The public checker verifies the released tables, expected fold coverage, historical-table linkage and checksums. It runs no model and cannot repeat the private prediction-level verification from the CSV files alone.

- Scientific identity: `fb59d7c6feab0a816833191c70d6a9c5884fa3233e25a3e94b3d781fff289aa9`.
- `fold_results.csv`: 2,000 newly fitted method/fold rows, five datasets.
- `aligned_20_task_ap.csv`: 20 dataset means, labelled historical or new. The file name predates the wording used here and is kept unchanged.
- `protocol.json`: immutable experiment settings, original source hashes, prepared-input/split hashes and job matrix.
- `completion.json`: portable verification summary without local paths.
- `verify_record.py`: standard-library stored-table and checksum checker, not a model rerun.
- `MANIFEST.sha256`: exact bytes of this supplement. Existing seven-file historical MANIFEST remains unchanged.
