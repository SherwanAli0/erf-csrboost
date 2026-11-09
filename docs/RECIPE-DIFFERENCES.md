# Recipe differences: live application versus the evaluation package

This page lists where the live FixYourDataset application (`backend/engine.py` and `backend/models/`) and the
evaluation package (`evaluation/`, import name `rare_event_eval`) do different things for the same named method.
It was written after reading both code bases. It does not claim that either side reproduces the research
tables, and it does not rank the recipes. The mapping of every legacy name to a package recipe lives in
`backend/services/recipes.py` and is checked by `tests/test_app_service.py`.

The application is unchanged at this stage. The package side describes what `backend/services/` calls.

## 1. Evaluation protocol

| Topic | Live application | Package and service |
|---|---|---|
| Partitions | One stratified split, 25 percent held out (`TEST_SIZE`), seed 42, made before any resampling. Uploads can also choose 3, 20 or 100 folds with training-row caps; predictions are pooled out of fold | One stratified split of about 60 / 20 / 20 train, validation and test, seed 42, with exact counts in the report. Other modes: group, exact-feature group, chronological. No repeated folds |
| Threshold | Fixed at 0.5 (`score >= 0.5`) for recall, precision, F1, G-Mean, balanced accuracy and the flagged-row list | Chosen on the validation partition from 101 fixed values (0.00 to 1.00), maximizing F1 or minimizing assumed cost, highest value on ties. Test labels are never used for the choice. The report records the value and its source |
| Test-set threshold slider | Part of the page; moves the cut on held-out probabilities | Not in the service. Planned only as a labelled exploratory view that cannot overwrite the validation-selected result |
| Training-row caps | Comparison trains on at most 4,000 stratified rows (`CAP_COMPARE`); fold runs use 8,000, 4,000 or 1,500 | No sampling. A table over the limits (default 20,000 rows) is refused with a typed error |

## 2. Metrics

Both sides call scikit-learn for the score-based metrics, so the values agree. `tests/test_app_service.py` asserts
equality on identical fixed predictions at several thresholds, and asserts each difference below.

| Metric | Result |
|---|---|
| Recall, precision, F1, balanced accuracy, accuracy, ROC AUC | Equal. The application rounds to four decimals, the package does not |
| `pr_auc` (application) and `average_precision` (package) | The same number: non-interpolated average precision. The application's name suggests the area under the precision-recall curve, which is a different quantity (trapezoidal area differs on the test fixture) |
| G-Mean | Reported by the application only. It equals the square root of recall times specificity for a binary target. The package reports specificity instead |
| Undefined values | The application substitutes zero (`zero_division=0`) or one (a single-class table gives recall 1.0). The package returns `null` with a warning |
| Threshold-free metrics on thresholded labels | Neither side does this: AP and AUC come from scores |

## 3. Random forest naming and settings

| Name | Settings |
|---|---|
| Application "plain" (`make_plain`, weighted) | `RandomForestClassifier`, 200 trees, `class_weight="balanced"`, `n_jobs=-1` |
| Package `rf_raw` | 100 trees, no class weights. This is the unweighted research baseline, the only recipe called plain or raw |
| Package `rf_weighted` | 100 trees, `class_weight="balanced"` |
| Package `rf_weighted_1000` | 1,000 trees, balanced |
| Package registry entry `legacy_rf_weighted_200` | Names the application recipe so reports can describe it. It is not runnable through the training comparison |

The application's weighted 200-tree forest is not the research 100-tree raw forest and must not be reported under that name.

## 4. ERF: bagged resampled forests, not boosting

Both sides use 20 bags of 50 trees (1,000 trees in total), a per-bag seed of `42 + bag * 7919`, mini-batch k-means
(`n_init=1`, `max_iter=100`, `batch_size=1024`) with k equal to the minority count, 50 percent retained per
cluster, and averaged probabilities. There is no boosting in ERF on either side. Differences:

- Preprocessing order. The application resamples the raw training rows, then fits its imputer, scaler and one-hot
  encoder on each resampled bag. The package fits the preprocessing once on the training partition and resamples
  inside the estimator.
- Categorical columns. The application clusters on ordinal codes (unscaled) next to standardized numeric columns and
  oversamples with SMOTE-NC on ordinal codes. The package's mixed variant clusters on standardized numeric columns plus
  0/1 indicator columns and oversamples with SMOTENC on integer codes; synthetic categories are neighbour modes, inside
  the training vocabulary.
- Failures. The application falls back to random oversampling when a sampler raises, and notes it in text. The package
  fails the model with `RESAMPLING_INFEASIBLE` (fewer than two rows to oversample), `RESAMPLING_NO_NUMERIC_FEATURES`,
  `FEATURE_EMPTY_CATEGORY_IN_TRAIN` (a categorical predictor with no observed training value),
  `RESAMPLING_CLUSTERING_FAILED` or `EXPANSION_EXCEEDED`.
- Parallelism: `n_jobs=-1` in the application, `n_jobs=1` in the package workers.
- Training rows: capped at 4,000 in the application, not capped in the package.

A recipe with fewer bags or trees would need its own recipe identifier and displayed capacity. None exists yet.

## 5. CSRBoost implementation differences

Three implementations carry the name. Read from the code:

| Item | Research comparison runner (`research/scripts/comparison_rf_csrboost_erf.py`) | Live application (`backend/models/csrboost_exact.py`) | Package (`evaluation/src/rare_event_eval/estimators.py`) |
|---|---|---|---|
| Clustering | Mini-batch k-means (`n_init=1`, `max_iter=100`) | Exact `KMeans(n_init=10)` | Exact `KMeans(n_init=10)` |
| Retention | 50 percent per cluster; draws only when it keeps fewer than the whole cluster | 50 percent per cluster; draws only when it keeps fewer than the whole cluster | 50 percent per cluster; always draws, so the random stream differs from the application's for the same seed |
| Base learner of the 50 AdaBoost rounds | `DecisionTreeClassifier(max_depth=1)`: depth-1 stumps | `DecisionTreeClassifier(random_state=seed)`: default, unlimited depth | `DecisionTreeClassifier(random_state=seed)`: default, unlimited depth |
| Learning rate, rounds | Library default (1.0), 50 | 1.0, 50 | Library default (1.0), 50 |
| Class-redefinition guard | None | Yes: labels are swapped if retention leaves the "rare" class larger | No guard. Class roles follow the counts of the training partition |
| SMOTE neighbours | k = min(5, max(1, n_min - 1)) | k = max(1, min(5, n_min - 1)) | k = min(5, retained minority - 1); fewer than two rows is an error |
| When SMOTE cannot run | Returns the unresampled data without any note | Returns the unresampled data without any note | Raises; the model is reported as failed (`strict` policy) |
| Feature space | Scaled numeric matrix | Dense matrix after one-hot encoding; SMOTE interpolates the 0/1 columns | Numeric pipeline (standardized), or the mixed variant with SMOTENC on integer codes |
| Training rows | All | At most 4,000 | All, within the table limits |
| Seeds | 42 | 42 for clustering, SMOTE, trees and AdaBoost | The job seed (default 42) for the same four |

So the application and the package use the same full-depth base learner, while the research runner used depth-1 stumps.
The application and package versions are re-implementations. None is shown to be the original authors' code, and none
reproduces the published CSRBoost numbers. The package recipe description states that it is not an exact paper
reproduction.

## 6. Preprocessing, labels and rows

| Topic | Live application | Package and service |
|---|---|---|
| Numeric columns | Median imputation and scaling | Same, fitted on the training partition only (`keep_empty_features=True`) |
| Categorical columns | Most-frequent imputation, one-hot with unknown values ignored | Training vocabulary, most-frequent training value for missing cells, one-hot indicators, unseen categories become an all-zero block. Cardinality and expanded-width limits are checked before any fit |
| Identifier-like columns | Object columns with more than half distinct values are dropped without asking (a warning is shown) | Never dropped silently. A high-cardinality predictor is refused (`TABLE_CARDINALITY_EXCEEDED`); the profile lists the legacy rule's columns as suggested exclusions that the caller confirms |
| Target column | Suggested and used | Suggested by the same function, flagged `suggested, needs confirmation`, then confirmed by the caller |
| Positive class | The rarest label, chosen silently | Explicit, required. The profile suggests the rarest label and flags it for confirmation |
| Labels | Case and spacing variants are merged; integral floats become integers; one-row classes are dropped with a warning | Exact label text; nothing merged or dropped. A one-row class is an error (`LABEL_SINGLETON_CLASS`) |
| More than two labels | The rarest label against all others, with a warning | Explicit `one_vs_rest`, otherwise `LABEL_MULTICLASS_REQUIRES_OVR` |
| Missing targets | Rows dropped with a warning | Error (`LABEL_MISSING_TARGET`) unless `missing_policy="drop_reported"`, which lists them |
| Unreadable or ragged rows | The legacy reader skips malformed rows and warns | The service keeps the legacy reader for uploads and copies its warnings and the skipped-row count into the report. The package's own CSV parser rejects malformed rows instead |

## 7. Not in the package yet

The leakage demonstration (leaky versus honest resampling with a 120-tree forest), the duplicate and single-column
proxy scan, "score new rows" on unlabelled data, the balanced-training-data and held-out CSV exports and the pipeline
snippet exist only in the live application. See `docs/CAPABILITY-MATRIX.md`.
