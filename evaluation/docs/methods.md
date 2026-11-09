# Method and execution notes

Author: Sherwan Ali.

ERF is a new explicit implementation of the project's experimental ensemble,
not a byte-identical reproduction of the historical runner. Its default is 20
bags of 50-tree random forests. Each bag clusters the majority class using
MiniBatchKMeans, keeps half of each cluster (rounded up), applies SMOTE to the
smaller retained class and fits a forest. Seeds and sampling counts are recorded.
No sampler exception is swallowed and no unresampled fallback is used.

CSRBoostClassifier implements the project's CSRBoost-style comparison with
KMeans, majority retention, SMOTE and 50 AdaBoost full-tree learners. Credit for
the source paper belongs to Yadav, Joshi, Mulye, Asari, Udmale and Bhole. Consult
the original replication audit for paper attribution and historical differences.
The new implementation does not claim exact paper reproduction.

All fresh comparisons use training-fitted median imputation and standardization.
RF references use 100 trees, except the explicitly named 1,000-tree capacity
reference. Balanced RF uses sampling_strategy=all, replacement=True and
bootstrap=False. Histogram gradient boosting uses 100 iterations without early
stopping. Logistic regression uses max_iter=2000. These are fixed settings,
not equally tuned models or equal actual compute.

The CLI caps data at 20,000 rows and 200 numeric predictors, with at least ten
rows and both classes in every outer partition. Model workers default to a
60-second wall-time limit and one numerical thread; CLI timeouts accept 1..600
seconds. The local web interface permits one training request at a time and up
to five models, each with a 60-second timeout. There is no hard operating-system
memory quota. Worker peak RSS includes interpreter/import overhead. Temporary
input files are local; an abnormal machine shutdown can leave temporary files.

The optional paired AP percentile bootstrap resamples paired rows or groups,
records skipped single-class draws, and is conditional on fitted models. It
does not capture training variation, adjust multiple comparisons or establish
significance. Temporal bootstrap intervals are rejected until a defensible block
design is supplied. The new pilot uses 200 draws and only two split seeds.

Saved-prediction evaluation cannot verify a user's declared test role or prevent
selection after inspecting outcomes. Fresh training enforces its own partitions,
but cannot detect every source of target leakage in supplied features.

## Mixed-type training adapter (application recipes)

`mixed.train_comparison_tabular` trains on tables with numeric, categorical and missing predictors. This is an
application adapter, not the numeric research recipe: nothing in it reproduces a research table, and the
mixed variants below have not been evaluated on the research datasets. A table whose predictors are all numeric
runs the unchanged numeric recipes and gives the same metrics as `train_comparison` for the same seed (a test
compares them to 1e-12). A table with at least one categorical predictor runs the mixed variants, which have
their own recipe ids in `registry.MIXED_REGISTRY` (`rare_event_eval/<model>-mixed/1`, `supports_mixed` true,
one entry per package model id, each exercised by the tests). Numeric recipe ids and records are unchanged.

Order of work: `profile_table` (byte, row, column, cardinality and one-hot width limits first), label
resolution with an explicit positive label (one-versus-rest is an explicit choice, missing targets are an error
unless `drop_reported` is requested and then listed), one outer split built in the parent with exact counts
(a split that cannot be built raises `SPLIT_INFEASIBLE`; nothing else is substituted), one subprocess per model
that fits preprocessing, resampling and calibration on the training partition only, a validation-selected
threshold, and test-partition evaluation. The worker receives typed columns (float arrays and `U` string arrays)
in an `.npz` written and read without pickle, plus a `config.json` of column kinds; there is no joblib or pickle
input and uploaded content is never executed.

Preprocessing, fitted on the training partition only:

- Numeric columns: median imputation (empty columns kept) and `StandardScaler`, as in the numeric recipes.
- Categorical columns: the sorted distinct training values are the vocabulary; missing cells (blank, `?`, `NA`,
  `NaN`, or a missing value in a DataFrame) take the most frequent training value, ties going to the smallest
  string. There is no target encoding.
- Unseen categories at validation, test or scoring time get a dedicated code (the vocabulary size) and expand to
  an all-zero indicator block for that column. Nothing is refitted. The count of unseen cells per partition is
  recorded.
- The final estimator sees the numeric columns followed by one 0/1 indicator column per vocabulary entry. The
  expanded width is bounded by `TabularLimits` (50 distinct values per categorical predictor and 400 expanded
  columns by default) before any model is fitted, and checked again on the training vocabulary inside the worker.
  The column order and a hash of the expanded names are recorded and identical for training, validation and test.

Resampling for categorical predictors (SMOTE-RF, CSRBoost-style, ERF and its ablations):

- Oversampling uses imbalanced-learn `SMOTENC` on a coded matrix: standardized numeric columns followed by one
  integer code per categorical column, which are declared categorical. A synthetic row takes the most frequent
  code among its neighbours, so it is always a member of the learned vocabulary. No fractional code is created
  or rounded; if a sampler ever returned one the model fails with `RESAMPLING_INVALID_CATEGORY_CODE`. The
  estimator then sees the indicator expansion of the resampled rows.
- Clustered majority retention clusters a different representation: the standardized numeric columns plus
  unweighted 0/1 indicator columns for each categorical column, fitted on the training rows. Categorical codes
  have no natural ordering, so they are never used as distances. This differs from the older FixYourDataset
  application, which clustered ordinal codes, and from the numeric research recipe, which has no categorical
  columns. Mini-batch k-means (ERF) and exact k-means (CSRBoost-style) keep the settings of the numeric recipes.
- Capacity, seeds and class weights come from the same constructors as the numeric recipes (the tests compare
  them). Only the representation and the oversampler differ.

Strict failures. `fallback_policy` has one value, `strict`. A model that cannot be built or fitted is reported as
`status: "failed"` with a stable `reason_code` and its message; the other models still report. Examples:
`RESAMPLING_INFEASIBLE` (fewer than two rows of the class to oversample), `RESAMPLING_NO_NUMERIC_FEATURES`
(`SMOTENC` needs a numeric predictor, so all-categorical tables fail the resampling recipes),
`FEATURE_EMPTY_CATEGORY_IN_TRAIN` (a categorical predictor with no observed value in the training partition has an empty
vocabulary, so `SMOTENC` cannot oversample; recipes without SMOTE still run), `RESAMPLING_CLUSTERING_FAILED`,
`EXPANSION_EXCEEDED`. If k-means finds fewer non-empty clusters than requested
(for example many duplicate rows), the run continues and the event records `actual_clusters` and a warning, as
the numeric recipes do; this is a disclosed property of the run, not a failure.

Calibration (`sigmoid`) uses three training-only inner folds with a feasibility check. Chronological splits with
sigmoid calibration are refused with `CALIBRATION_INCOMPATIBLE`, because ordered inner folds can separate equal
timestamps. The numeric 0.3.0 path still accepts that combination with its documented limit; the adapter does not.
Combined group and chronological splitting is refused with `SPLIT_UNSUPPORTED`.

Limitations. The recipes were exercised on small synthetic tables in tests; their behavior on large real tables,
or their accuracy, is not measured here. Rows are held in an object matrix inside the worker, which uses more
memory per cell than a float matrix. The cardinality and width limits are local defaults, not verified limits of
any hosted deployment. Indicator expansion of high-cardinality columns is bounded, not solved: an identifier
column must be excluded explicitly (`id_columns`, `exclude_columns`).
