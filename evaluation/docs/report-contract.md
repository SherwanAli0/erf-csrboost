# Report contract, schema 1.0

Author: Sherwan Ali. The API returns a standard JSON-compatible object. Nonfinite
JSON constants such as `NaN` and `Infinity` are not permitted. Unavailable metrics
use `null`. No automatic winner, learned policy or confidence interval is returned.

| Field | Meaning |
|---|---|
| `schema_version` | The structural contract, currently `1.0`. |
| `tool` | Package, Python, NumPy and scikit-learn versions. |
| `created_utc` | UTC execution timestamp. Results are deterministic for fixed input/configuration; this timestamp is not. |
| `task` | Exact class labels, binary scope, score kind, user-declared evaluation role and `role_verified: false`. |
| `data` | Evaluated row counts, positive prevalence, zero dropped rows and optional user-declared description; CLI additionally records the original CSV SHA-256. |
| `policies` | Fixed threshold, review budgets, tie rule and optional cost assumptions. |
| `checks` | Input validation performed; training leakage, split overlap, independence and uncertainty not established. |
| `models` | Mapping from score-column/model name to metrics, fixed-policy outcomes, calibration bins and warnings. |
| `limitations` | Explicit interpretation limits carried into JSON and HTML. |

## Score semantics

All scores must be aligned with the supplied labels in input row order and
oriented toward the declared positive class. Those provenance claims cannot be
verified from a CSV of predictions alone. Probability scores must be finite in
`[0, 1]`; ranking scores may be any finite values. Scores are never silently
inverted, clipped or normalized.

`average_precision` is scikit-learn's non-interpolated recall-increment weighted
average of precision. `roc_auc` measures ranking discrimination. This release
returns `null` for both when the evaluation contains only one class.

`brier_score` is mean squared error of the supplied positive-class probability
against the 0/1 target; binary range `[0, 1]`. `log_loss` uses scikit-learn's
natural-log loss with its documented numerical endpoint clipping. These are
available only for probability scores. Lower is better. Neither value alone
certifies calibration. The input scores themselves are not changed by clipping.

## Policy outcomes

The threshold rule is `score >= threshold`. A review budget selects the highest
`k` scores. Stable descending score order preserves original row order within
ties. Exactly `k` rows are flagged; `k=0` and `k=n` are supported.

Confusion counts use the declared positive class. Precision is `TP/(TP+FP)`,
recall is `TP/(TP+FN)`, specificity is `TN/(TN+FP)`, false positive rate is
`FP/(TN+FP)`, F1 is `2TP/(2TP+FP+FN)`, and accuracy is `(TP+TN)/n`.
Balanced accuracy is the mean of recall and specificity when both are defined.
Every zero denominator produces `null`.

At a cutoff that divides an exact score tie, the report records tie size,
remaining review slots, the actual result using input order and possible
minimum/maximum true positives, precision and recall. These bounds allocate
the remaining slots in every possible way within the tie. They are descriptive
possibility bounds and say nothing about sampling uncertainty.

Cost assumptions are nonnegative, finite numbers in the same user-defined unit.
The score is `FP * false_positive_cost + FN * false_negative_cost + flagged * review_cost`.
False alarms therefore incur both review and false-positive costs. The two error
costs must be supplied together. Nonfinite calculated totals cause an error.

## Descriptive reliability bins

Probability scores are assigned to equal-width bins over `[0, 1]`. Intervals are
closed on the left and open on the right, except the final bin includes 1.
All bins are returned, including empty bins. Each contains its row count, mean
score and observed positive fraction. No bin-based calibration error, fitted
calibrator or calibration-validity claim is produced.

## Scope and privacy

Reports export aggregates, class names, model names, optional user-declared descriptions and configuration. They do
not export per-row scores, labels, input paths, identifiers or fitted models.
Aggregate counts and user-supplied names can nevertheless be sensitive.
No raw-file checksum is added by the Python API, which accepts arrays; the CLI
can checksum the exact CSV bytes it reads.

Selecting policies or models after inspecting test outcomes is exploratory.
Nothing in this contract establishes independence, a leak-free training process,
causal utility, statistical significance or performance on another population.

## Training comparison report, schema `training-1.1`

`train_comparison` now returns `schema_version: "training-1.1"`. The change is additive:
every `training-1.0` field keeps its meaning, and `training_report` and `render_html` still
read `training-1.0` results (a result without the new fields is rendered without a
`training_provenance` block). A schema string they do not know is refused.

| Field | Meaning |
|---|---|
| `label_mapping` | Mode (`binary` or `one_vs_rest`), the explicit positive label, negative label(s), every original label with its count and internal value (in `one_vs_rest` mode each other label merged into the negative class appears with internal value 0, singleton labels included, so no merge is silent), input/missing/effective row counts, the missing-target policy, listed excluded rows (first 100, with a truncation flag) and the probability orientation. The positive label is never inferred. |
| `predictors` | Feature names used as predictors, or `null` when the caller supplied bare arrays. |
| `excluded_columns` | Target, group, time and identifier columns with their roles. They are never predictors. |
| `input_parsing` | Parser name, optional parser record (encoding, delimiter, hash, rows read/accepted/rejected), rows in, rows used, rows excluded and the listed exclusions. |
| `split` | `requested` (mode, seed, approximate target fractions) and `achieved` (mode, exact rows and positive/negative counts per partition, index hash, `fallback_used: false`, whether all models share the partitions). `achieved` is `null` if no model completed. |
| `model_records` | Per requested model: id, display name, record version, `recipe_id`, family, declared parameters with the job seed resolved, `supports_mixed`, description, `cost_class` (qualitative, not measured) and scope. Each outcome also carries the same `model_record`; the constructor parameters actually built are in `execution.model_parameters`. |
| `preprocessing` | The shared numeric preprocessing and its fitting scope; `mixed_type_support` is false for these recipes. |
| `fallback_policy` | `policy: "strict"` (the only policy implemented) and `fallback_used: false`. A model that cannot be built or fitted is reported, never replaced. |
| `failures` | One entry per non-successful model with a stable `reason_code` and the message. Outcomes also carry `reason_code`. |
| `seed` | The job seed. |

Failure reason codes: `MODEL_TIMEOUT`, `WORKER_NO_RESULT`, `MODEL_UNKNOWN`,
`MODEL_DEPENDENCY_MISSING`, `CALIBRATION_INFEASIBLE`, `RESAMPLING_INFEASIBLE`,
`SPLIT_INFEASIBLE`, `INPUT_INVALID`, `MODEL_OUT_OF_MEMORY`, `MODEL_FIT_ERROR`. The worker reports an
exception message and the code is assigned from it, so a reworded library message can fall
through to `MODEL_FIT_ERROR`; the message is always kept beside the code.

### Labels (`rare_event_eval.labels`)

`resolve_labels(values, LabelSpec(positive, negative=None, mode="binary"), missing_policy="error")`
returns internal 0/1 values for the retained rows, the mapping record above and the exclusion list.
Failures raise `LabelError` (a `ValueError`) with a stable `code`: `LABEL_POSITIVE_REQUIRED`,
`LABEL_POSITIVE_ABSENT`, `LABEL_NEGATIVE_ABSENT`, `LABEL_MISSING_TARGET`, `LABEL_SINGLE_CLASS`,
`LABEL_SINGLETON_CLASS`, `LABEL_MULTICLASS_REQUIRES_OVR`, `LABEL_SPEC_INVALID`. Binary mode needs
exactly two distinct labels; more need `mode="one_vs_rest"`, which merges the other labels into the
negative class and records them. Missing means `None`, NaN-like values and blank strings only; text such
as `NA` is a real label. `missing_policy="drop_reported"` excludes missing-target rows and lists them with
counts. Singleton and single-class targets are never repaired. `train_comparison` resolves labels before
any worker starts and raises `LabelError` instead of reporting a per-model failure.

### Tables (`rare_event_eval.tabular`)

`parse_table_csv` and `profile_table` handle mixed-type tables with the standard library (pandas is an
optional `app` extra, used only by `table_to_dataframe`; DataFrames can be profiled without it). The
numeric-only `datasets.parse_training_csv` is unchanged. Encoding is strict UTF-8 (a BOM is recorded and
removed); other encodings must be named, and no fallback encoding is guessed. The delimiter is chosen from
comma, semicolon, tab and pipe by a field-count scan of the first 50 lines, and the decision, candidates
and an ambiguity flag are recorded. A row with the wrong field count, a blank line inside the data or a
quoting error is rejected with `TABLE_MALFORMED_ROW`; trailing blank lines are counted. Target, group,
time and identifier columns are listed as excluded metadata. No other column and no row is dropped;
constant, empty, duplicate and high-cardinality predictors are reported.

Default limits (`TabularLimits`) are conservative local defaults for this package, not verified limits of
any hosted deployment: 5,000,000 bytes, 20,000 rows, 205 columns, 50 distinct values per categorical
predictor, and an estimated 400 one-hot predictor columns. They raise `TABLE_TOO_LARGE_BYTES`,
`TABLE_TOO_MANY_ROWS`, `TABLE_TOO_MANY_COLUMNS`, `TABLE_CARDINALITY_EXCEEDED` and
`TABLE_EXPANSION_EXCEEDED`, in that order of cost: bytes before decoding, columns before any row is read,
rows as soon as the count is exceeded, cardinality while counting distinct values (counting stops at
limit plus one), and expansion from the width estimate before any encoding. The profile's one-hot width is
an upper bound (distinct non-missing values across the whole table).

### Model registry (`rare_event_eval.registry`)

`REGISTRY` maps stable model ids to `ModelSpec` records. `MODEL_NAMES` and `DEFAULT_MODELS` in
`training.py` are derived from it and keep their original ids and order. `describe_model(id)` returns the
record plus the parameters read from the object `make_model` builds (nothing is fitted), and the tests
compare the two. What the recipes construct: `rf_raw` is an unweighted 100-tree forest; `rf_weighted` and
`rf_weighted_1000` use `class_weight="balanced"` with 100 and 1000 trees; `logistic` and
`hist_gradient_boosting` are unweighted; `erf` is 20 bags x 50 trees (bagged resampled forests, not
boosting) and `erf_one_bag`, `erf_no_clustering`, `erf_no_smote` are named ablations. The registry also lists
`legacy_rf_weighted_200` (scope `legacy_app`, 200 trees, class-weighted), documented as the legacy
FixYourDataset app recipe; it is not the research raw forest, is not in `MODEL_NAMES` and cannot be run
through the training comparison. `supports_mixed` is false for every numeric recipe in `REGISTRY`. The mixed-type
variants are separate entries in `MIXED_REGISTRY` (`scope: package_mixed`, ids ending in `-mixed`, own `recipe_id`,
`supports_mixed` true, `variant_of` naming the numeric recipe). `describe_mixed_model(id)` reads the parameters of the
object `make_mixed_model` builds, and the tests compare it with the declared record.

### Mixed-type adapter results (`rare_event_eval.mixed`)

`train_comparison_tabular(table_or_csv_bytes, *, target, label_spec, predictors=None, exclude_columns=(), group_column=None,
time_column=None, id_columns=(), split_mode="stratified", seed=42, models=DEFAULT_MODELS, calibration="none",
limits=TabularLimits(), timeout_seconds=60, missing_policy="error", fallback_policy="strict", review_fractions=(.1, .2),
costs=None, description=None, ...)` returns a `training-1.1` result with the fields above and these additions or
differences. The method is described in [methods.md](methods.md).

| Field | Meaning |
|---|---|
| `predictor_kinds` | Per predictor: `numeric` or `categorical`. |
| `excluded_columns` | Target, group, time and id columns (as in `profile_table`, with `used_by_split` for group and time), plus columns not in the selected predictor list (`role: not_selected`) or excluded by the caller (`role: excluded_by_user`). |
| `table_profile` | The `table-profile-1.0` record without the parsing, predictor and exclusion blocks that are reported separately. |
| `split.achieved` | Built once in the parent, so `identical_across_models` is true and `built_once_in_the_parent` is true. An infeasible request raises `SPLIT_INFEASIBLE` instead of producing a result. |
| `preprocessing.adapter` | Adapter version, `recipe_variant` (`numeric` or `mixed`), the recipe rule, unseen-category rule, clustering representation and oversampler. |
| `preprocessing.record` | Fit scope, per-predictor kind, imputation and missing-cell counts per partition, per categorical column the vocabulary size, most frequent training value, indicator column names and unseen-cell counts per partition, the expanded feature names and count, and the alignment record (partition shapes, identical columns across partitions, a hash of the column names). Category names appear in this record and in the HTML provenance, where they are escaped. |
| Outcome `status` | `success`, `failed` (worker error; the raw worker status is kept as `worker_status`) or `timeout`. The numeric path still reports `error` for a worker error. |
| Outcome `execution` | Adds `recipe_variant`, `preprocessing_record`, `constructed_steps` (class and parameters of every step) and `resampling_events`; for categorical resampling each event names `smote_variant` (`SMOTENC`), `synthetic_rows` and `synthetic_category_codes_in_vocabulary`. |

Whole-request errors are typed `ValueError` subclasses with a stable `code` and are raised before any worker starts:
`TabularError` (`TABLE_*`, including `TABLE_CARDINALITY_EXCEEDED` and `TABLE_EXPANSION_EXCEEDED`), `LabelError`
(`LABEL_*`) and `MixedError` with `OPTION_INVALID`, `PREDICTOR_UNKNOWN`, `NON_FINITE_VALUE`, `GROUP_MISSING`,
`TIME_INVALID`, `SPLIT_INFEASIBLE`, `SPLIT_UNSUPPORTED`, `CALIBRATION_INCOMPATIBLE`, `CALIBRATION_INFEASIBLE`,
`CATEGORICAL_REJECTED` and `PANDAS_REQUIRED`. Model failure reason codes add `RESAMPLING_NO_NUMERIC_FEATURES`,
`FEATURE_EMPTY_CATEGORY_IN_TRAIN` (a categorical predictor with no observed training value; SMOTENC cannot build its
vocabulary), `RESAMPLING_CLUSTERING_FAILED`, `RESAMPLING_INVALID_CATEGORY_CODE` (a sampler returned a code outside a
non-empty vocabulary) and `EXPANSION_EXCEEDED` to the list above; a
code attached to the exception in the worker takes precedence over the message classification.

The `train` command uses the adapter when the CSV has a non-numeric predictor or when `--predictor`, `--exclude-column`,
`--id-column`, `--one-vs-rest`, `--missing-target drop_reported`, `--max-categories` or `--max-expanded-features` is given;
otherwise the numeric parser and recipes run as in 0.3.0. A categorical predictor needs pandas (extra `[app]`); the CLI
reports a clear error without it. `--categorical reject` keeps the numeric-only behavior.

## Primary references

- [Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
- [Log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
- [Threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [Calibration](https://scikit-learn.org/stable/modules/calibration.html)
