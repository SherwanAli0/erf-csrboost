# November 2026 ERF evaluation record

Author: Sherwan Ali. Updated 7 November 2026.

The five-dataset extension completed 2,000 new fits: five datasets x four methods x 100 folds. All succeeded, with zero errors, timeouts or recorded SMOTE fallbacks. These new results are combined descriptively with the original 15-dataset June record. This is not a fresh rerun of all 20 datasets in one historical software environment.

## Main result

Mean average precision (AP) is compared after averaging the 100 folds within each dataset. Win/tie/loss counts are descriptive; differences within 1e-12 are treated as numerical ties. They are not significance tests.

| Baseline | ERF higher AP | Numerical ties | ERF lower AP | Mean dataset AP difference |
|---|---:|---:|---:|---:|
| CSRBOOST | 16 | 0 | 4 | +0.068523 |
| RF_BRF | 16 | 0 | 4 | +0.009251 |
| RF_RAW | 12 | 2 | 6 | -0.000777 |

ERF is strictly highest among these four methods on seven datasets and tied for highest on two more. Its average dataset AP is slightly below ordinary Random Forest despite winning more individual datasets. This does not establish ERF as the best overall method.

The updated comparison is 16/20 higher-AP datasets versus the local CSRBoost reimplementation: 12/15 historical datasets plus 4/5 newly aligned datasets. The old 17/20 statement combined different protocols and remains retired. The 18-dataset July record remains a separate exploratory record. The small coil loss (-0.0001196 AP) is a descriptive numerical loss, not established meaningful inferiority.

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
- ERF: 20 independently resampled bags, 50 RF trees per bag; bag seeds 42 + 7919*b. Raw RF and Balanced RF each use 100 trees. These are not equal-capacity comparisons.
- One computational thread. Historical resampling fallback behavior is preserved but instrumented; no fallback occurred in the 2,000 new fits.
- Metrics: accuracy, ROC AUC, F1, AP and G-mean. Full new fold metrics, including failed-cell status fields, are in fold_results.csv; there are no failed cells in this completed record.
- Python 3.12 laptop and cloud execution with pinned dependencies: numpy 2.2.6, scipy 1.18.1, pandas 2.3.3, scikit-learn 1.8.0, imbalanced-learn 0.14.1, joblib 1.6.0, threadpoolctl 3.8.0. Historical June dependency versions are not fully known.
- Laptop: mammography and thyroid_sick, 800 fits. Colab: coil_2000, satimage and creditcard, 400 fits each. Kaggle was not used for this five-dataset extension.

| Added dataset | Rows | Features | Positive rows |
|---|---:|---:|---:|
| mammography | 11183 | 6 | 260 |
| thyroid_sick | 3772 | 52 | 231 |
| coil_2000 | 9822 | 85 | 586 |
| satimage | 6435 | 36 | 626 |
| creditcard | 14992 | 30 | 492 |

Creditcard is the historical 14,992-row subset: all 492 fraud rows plus 14,500 nonfraud rows selected without replacement with RandomState(42), then concatenated/shuffled using the preserved preparation. It is not the full credit-card source dataset. Prepared artifact and split hashes are recorded in protocol.json. Dataset arrays and row-level predictions are retained privately; they are not redistributed in this result supplement. The public supplement verifies stored arithmetic, not full regeneration without those inputs.

## Cost and interpretation

| Added dataset | Median paired ERF / raw-RF fit-and-score time |
|---|---:|
| coil_2000 | 25.31x |
| creditcard | 6.60x |
| mammography | 14.91x |
| satimage | 35.59x |
| thyroid_sick | 17.36x |

Timing ratios pair the same folds within each dataset on its assigned machine. They are observed wall-clock fit-and-score costs, not controlled compute-budget benchmarks or cross-hardware performance claims. The imbalance between 1,000 ERF trees and 100-tree forest baselines prevents attributing gains to the sampling mechanism alone. Next research direction: equal tree/runtime budgets and component ablations; those experiments have not been run.

The original data caveats remain: duplicate and train/test feature twins, conflicting labels, shared Car and Yeast feature collections, rare positives and the 196-row Glass variant. Twenty dataset labels represent 18 bookkeeping feature collections, not 20 independent data sources. Dataset reuse during development remains exploratory. No pooled-fold p-value, universal superiority or publication claim is made. Historical June AUC findings are still valid as that separate record; they must not be relabelled as the new AP result. The app has its own preprocessing and evaluation and was not changed or redeployed for this release.

## Separate adaptive campaign

A separate 480-cell campaign evaluated fixed ERF, adaptive sampling, adaptive ensemble size and their combination. It completed with 382 newly executed successes, 96 reused results and two explicit errors; zero pending/timeouts. The two failures concern YEAST5ERL feature-group inner splits lacking both classes. Its primary recall-at-10%-review comparison is against fixed ERF, not CSRBoost: on stratified two-seed dataset means, adaptive sampling gave 6 higher / 12 tied / 2 lower datasets, adaptive size 3 / 12 / 5, and both 5 / 10 / 5. This is exploratory, uses a different protocol and is not pooled with the 2,000-fit AP extension. Full adaptive source/results remain a separate local record; this supplement does not claim to release or independently regenerate that campaign.

## Verification and scope

The local completion audit checked all 2,000 job identities, frozen source/data/split hashes, ZIP CRCs, prediction hashes, exact test indices/labels, population sizes and recomputed all five metrics from every saved prediction artifact within 1e-12 tolerance. Original final archives remain preserved locally. The public checker verifies the released tables, expected fold coverage, historical-table linkage and checksums; it cannot repeat private prediction-level verification from CSVs alone.

- Scientific identity: `fb59d7c6feab0a816833191c70d6a9c5884fa3233e25a3e94b3d781fff289aa9`.
- `fold_results.csv`: 2,000 newly fitted method/fold rows, five datasets.
- `aligned_20_task_ap.csv`: 20 dataset means, explicitly labelled historical or new.
- `protocol.json`: immutable experiment settings, original source hashes, prepared-input/split hashes and job matrix.
- `completion.json`: portable verification summary without local paths.
- `verify_record.py`: standard-library stored-table and checksum checker, not a model rerun.
- `MANIFEST.sha256`: exact bytes of this supplement. Existing seven-file historical MANIFEST remains unchanged.
