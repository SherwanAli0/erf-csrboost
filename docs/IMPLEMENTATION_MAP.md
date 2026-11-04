# Implementation map: research runner and web application

ERF-CSRBoost exists here in two implementations.

- The **research runner**, `research/scripts/comparison_rf_csrboost_erf.py`. It produced the stored
  result tables (see `RESULTS_EVIDENCE.md`).
- The **application**, `backend/models/erf_csrboost.py` together with `backend/engine.py`. It
  serves the live app at https://fixyourdataset.fly.dev.

The application's ERF is **adapted from** the research implementation. It is **not numerically
identical** to it, and the result tables describe the research runner, not the application.

ERF-CSRBoost is a bagged ensemble of resampled random forests. It is not a boosting method.

Status labels used below: "read from code" means a person read both implementations. "Newly
executed" means a small check was run during release preparation with scratch scripts that are not
part of this repository (so a reader cannot repeat that exact check from the repository).

## 1. What ERF-CSRBoost does (research version)

For each of 20 bags, with bag seed `42 + bag * 7919`:

1. Cluster the majority-class training rows with MiniBatchKMeans, using K equal to the number of
   minority rows (n_init 1, max_iter 100, batch size 1024).
2. Keep 50 percent of each cluster (rounded up, at least one row).
3. Add the minority rows and run SMOTE (k = min(5, minority rows minus 1)).
4. Fit a 50-tree RandomForest on the rebalanced bag.

The 20 bags' predicted probabilities are averaged. The decision threshold is 0.5. In total that is
20 x 50 = 1,000 fully grown trees plus 20 clusterings and 20 SMOTE passes per fit.

Relationship to CSRBoost: the front end (clustered undersampling, then SMOTE) follows the recipe of
the published CSRBoost method, but it is "CSRBoost-style", not the published step in every detail.
The published step, as implemented in the separate audit repository, uses KMeans with n_init 10 and
a class-redefinition guard that swaps labels if the retained minority exceeds the retained
majority. ERF uses MiniBatchKMeans with n_init 1 and has no guard, in the research runner and in the
app. This was compared against the audit repository's code, not against the paper text.

## 2. Map: research runner against application

Research code is cited by function name in `comparison_rf_csrboost_erf.py` (definition lines in the
selected copy are given in section 3). Application code is cited by function name in
`backend/engine.py` and `backend/models/erf_csrboost.py`. Function names are used instead of line
numbers because line numbers move when files are edited.

| Aspect | Research runner | Application | Same or different | Evidence (read from code unless stated) |
|---|---|---|---|---|
| Bag and tree counts | 20 bags (`N_BAGS`), 50 trees (`N_TREES`) | 20 bags, 50 trees (`N_BAGS`, `N_TREES`) | Same | constants in both files |
| Per-bag seed | `rs = SEED + b * 7919`, SEED 42; used for clustering, retention, SMOTE and the forest | `SEED + bag * 7919`, SEED 42; used for the same four steps | Same | `run_erf_csrboost`; `train_erf`, `_make_sampler_original`, `_clustered_retain` |
| Clustering step | MiniBatchKMeans(K, n_init=1, max_iter=100, batch_size=1024), K = min(n_min, n_maj), keep ceil(50 percent) per cluster | Same parameters, same K, same retention rule | Same | `_cluster_k`, `_cluster_retain`; `_clustered_retain` |
| Feature space for clustering | StandardScaler fitted on the training fold (both classes) and applied to the whole matrix first | StandardScaler fitted on the training split after median imputation; categorical columns ordinal-encoded and appended unscaled | Same for complete all-numeric data; different when categorical columns or missing values exist | `run_experiment` fold loop; `resample_train` |
| SMOTE neighbours | k = min(5, max(1, n_pos minus 1)), default SMOTE, random_state = bag seed | k = max(1, min(5, minority count minus 1)); SMOTE-NC if categorical columns exist | Same parameters | `_apply_smote`; `_make_sampler_original` |
| SMOTE feature space | Works on the matrix already standardised with the training-fold scaler | Fits a new StandardScaler on the retained subset only, runs SMOTE, then converts back to original units | Different (the synthetic points differ) | `_apply_smote`; `resample_train` |
| SMOTE failure fallback | Any exception: use the retained set without oversampling | Any exception: random oversampling of the retained set, with a note to the user | Different | `_apply_smote`; `resample_train` |
| Class-redefinition guard of the published method | Absent | Absent in ERF (present only in the app's separate CSRBoost comparator) | Same (both lack it) | `run_erf_csrboost`; `train_erf`; `csrboost_exact.py` |
| Classifier in a bag | RandomForestClassifier(50 trees, random_state = bag seed, n_jobs=-1), other defaults, no class weights | Same estimator inside a scikit-learn Pipeline | Same | `_build_rf_erf`; `train_erf` |
| Input to the forest | Standardised matrix straight from SMOTE | Pipeline preprocessor re-fitted on the resampled data (median imputer and StandardScaler for numeric, most-frequent imputer and one-hot for categorical) | Different mechanics. Forests are invariant to per-column scaling, so for numeric data only the SMOTE-space difference above matters | `run_erf_csrboost`; `_preprocessor` in the engine |
| Missing values | Loader coerces to numeric and fills with 0 before any split (label free) | Median or mode fill fitted on training rows, plus pipeline imputers | Different (app extension) | `_to_float`; `resample_train` |
| Categorical features | Dataset-specific ordinal maps or dummy coding applied by the loaders to the whole dataset; no SMOTE-NC | One-hot in the pipeline, ordinal for clustering, SMOTE-NC | Different (app extension) | loaders; `_make_sampler_original` |
| Positive class | The loader makes the minority class label 1. The corrected runner flips the Sonar (CB) polarity; the run behind the stored table did not (see `RESULTS_EVIDENCE.md`) | `prepare` orders classes majority first so label 1 is the rarest class | Same intent | loaders; `prepare` |
| Multiclass targets | Not applicable: every loader binarises | Rarest class against all others, with a warning | Different (app extension) | `prepare` |
| Probability alignment | `predict_proba(...)[:, 1]` per bag, mean over bags, assumes classes_ = [0, 1] | `predict_proba(X)[:, 1]` per bag, mean; returns [1 minus p, p]; all 20 bags had classes_ = [0, 1] in the check | Same | `run_erf_csrboost`; `ERFModel.predict_proba`; newly executed |
| Threshold | 0.5 with `>=` for ACC, F1, G-Mean; AUC and AP from probabilities | `predict` uses `>= 0.5`; the UI also has a threshold slider that moves the cut on the held-out probabilities | Same default; the slider is an app addition and is exploratory | `_metrics`; `ERFModel.predict` |
| Metrics | acc, auc, f1, ap, gmean | recall, precision, f1, gmean, balanced accuracy, accuracy, pr_auc, roc_auc | Different set, same definitions where they overlap | `_metrics`; `classify_compare` |
| Evaluation protocol | RepeatedStratifiedKFold(5 splits, 20 repeats, random_state 42): 100 folds per dataset | Default: one stratified 75/25 split (random_state 42). Optional 3, 20 or 100 folds with row caps | Different. The app default is a demo-scale single split | `run_experiment`; `classify_compare`, `cross_val_compare` |
| Training size | No cap | Training split capped at 4,000 stratified rows for the model comparison; the fold options cap training rows at 8,000, 4,000 and 1,500 for 3, 20 and 100 folds | Different (app extension) | `_cap_train`; `cross_val_compare` |
| CSRBoost comparator | Live CSRBoost with MiniBatchKMeans and depth-1 stumps (AdaBoost, 50), seed 42 in every fold, no guard | `backend/models/csrboost_exact.py`: KMeans(n_init=10), default-depth tree base learner, class-redefinition guard | Different from the research comparator; the app's version follows the audit repository's function | `run_csrboost`; `csrboost_exact.py` |
| Plain baseline | RF-Raw: 100 trees, no class weights, scaling only | `make_plain`: 200 trees, balanced class weights | Different baselines | `_build_rf_raw`; `plain_model.py` |
| Errors | A method that raises becomes NaN and is dropped from the means (the stored runs contain no NaN) | Exceptions are caught and reported in the result note | Different | `run_experiment`; `classify_compare` |

## 3. Selected scripts and hashes

SHA-256 of the files exactly as stored. The research files were saved with Windows line endings
(CRLF). Git may store them with LF endings, so the hash of a cloned copy can differ from the hash
of the stored bytes. Both are given; they identify the same text.

| File | Role | SHA-256 of stored bytes | SHA-256 with LF endings |
|---|---|---|---|
| `research/scripts/comparison_rf_csrboost_erf.py` | Selected research runner (corrected loaders) | `3a799cfa3fcc53e4e3cb6df8fc94fc8ebd53f4d409dad0ea246615ba838f42bd` | `e2eef829d96f62efdfeac7cb162ca5932bfd8c5cd0bbb25494b11f18f0e11fcf` |

Definition lines in that file: `_to_float` 220, `_load_psdas` 246, `_load_esr` 256, `_load_dccc` 268,
`_load_cb` 299, `_metrics` 430, `_cluster_k` 444, `_apply_smote` 448, `_cluster_retain` 457,
`_build_adaboost` 476, `_build_rf_erf` 490, `_build_rf_raw` 498, `_build_rf_brf` 525,
`run_csrboost` 556, `run_erf_csrboost` 583, `run_rf_raw` 604, `run_rf_brf` 618, `run_experiment`
680, `main` 953.

Application files at the time of writing (SHA-256 of the committed text, LF endings):

| File | SHA-256 |
|---|---|
| `backend/models/erf_csrboost.py` | `11b4c061d8e1e89bff62b61e569361fe50fc491fed88905e7254d186db840d0a` |
| `backend/engine.py` | `f47f7cddbe545b02fc9d1af6e59f2b701a5723dad663fdcffc1027e9c3b9ec6d` |
| `backend/models/csrboost_exact.py` | `e29d08712dca35342ccd1d4d3e2321696e4847208f3995d9138ccbd4a6fa2de5` |
| `backend/models/plain_model.py` | `2176af4acb4a9ffd9acbf446ece523c39e23f8a140793c9f91e892a3eacffc80` |

The hash table is a snapshot. If a file changes, `git log` and `git diff` are the record of what
changed, and the hash here no longer applies.

### Which script produced the stored results

The stored April 2026 results (`RESULTS_EVIDENCE.md`) were produced by an earlier version of the
runner. That earlier script text was not archived. A later snapshot of the script (SHA-256
`87445debd3514c96b1c43aa1693d0cf44472721f4f70507a8a3b6a782bde5dd9`, dated 4 May 2026) is
consistent with it. It differs from the selected runner above in three places, none in the ERF
algorithm:

- the ESR and DCCC loaders pointed at folder names that do not exist in the final layout, so those
  two datasets would be skipped with a printed message;
- the Sonar (CB) loader made the larger class (M, 111 rows) the positive class;
- a docstring on `_cluster_retain` still said KMeans.

The ERF functions (`_cluster_k`, `_apply_smote`, `_cluster_retain`, `_build_rf_erf`,
`run_erf_csrboost`) are syntactically identical (same AST, ignoring docstrings) between the
snapshot and the selected runner. That is circumstantial support that the April run used the same
algorithm. It is not proof, because the April script itself is not preserved. The package versions
(Python, scikit-learn, imbalanced-learn, NumPy) used for the April run were not recorded.

## 4. A small equivalence check (newly executed, synthetic data)

One check compared the research-style pipeline with the application's `train_erf` on the same
synthetic dataset (900 rows, 10 numeric features, about 10 percent positive, one fixed seed). The
check ran in scratch space with the research functions extracted by syntax tree, so the research
script was not imported or run. Environment: Python 3.11.9, NumPy 2.4.6, pandas 2.3.3,
scikit-learn 1.8.0, imbalanced-learn 0.14.1, SciPy 1.16.3.

| Observation | Result |
|---|---|
| Bag 0 resampled size and class counts | Research 648 rows (324 and 324); application 648 rows (324 and 324) |
| Retained original rows before SMOTE | Identical (largest absolute difference 8.9e-16) |
| Synthetic SMOTE rows | Same count, different coordinates (largest difference 2.44 in original units, mean 0.08), because of the SMOTE-space scaler difference |
| 20-bag probabilities on the same 225 test rows | Mean absolute difference 0.0124, maximum 0.0700, Spearman 0.9959, same 0.5 label on 96.9 percent of rows |
| AUC and AP on this one split | Research-style 0.8420 and 0.5756; application 0.8345 and 0.5536 |
| Application determinism | Two `train_erf` calls on identical input gave identical probabilities |
| Training split with exactly one minority row | Research: SMOTE fails, the 30 majority rows and 1 minority row are kept. Application: 30 and 30 after random oversampling |

Conclusion: for complete numeric data the selection of retained rows matches, while the synthetic
points and the tiny-class fallback differ. The application is close to, but not numerically equal
to, the research ERF. This one-split synthetic check reproduces no stored result and supports no
performance claim.

## 5. What this map does not establish

- It does not show that the application reproduces any number in `RESULTS_EVIDENCE.md`. It does
  not, and it is not meant to.
- It compares one version of the research runner, not every historical copy. Earlier exploratory
  scripts (for example a variant that fits a balanced random forest inside each bag) are not
  ERF-CSRBoost and are not part of this release.
- It compares the front end with the audit repository's implementation of the published method,
  not with the paper text.
