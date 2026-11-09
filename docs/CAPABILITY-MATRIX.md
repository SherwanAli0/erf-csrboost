# Capability matrix: the live application and its planned successor

Every capability of the live FixYourDataset application, taken from an inspection of `backend/server.py`,
`backend/engine.py`, `backend/track.py` and `frontend/`, with what is planned for it. Nothing in the live
application has changed yet; the new code is a service layer (`backend/services/`) and tests that call the evaluation
package beside it. Recipe-level differences are in `docs/RECIPE-DIFFERENCES.md`.

Dispositions:

- **retained**: the capability stays, possibly behind a thin compatibility adapter.
- **redesigned**: the capability stays with deliberately changed behavior, for the reason given.
- **pending**: not designed or not built yet; no claim is made that it will keep its current form.

"Now" says what exists after this stage: **live** (the unchanged application only), **service** (also available through
`backend/services/` and covered by `tests/test_app_service.py`), or **package** (in `evaluation/`, not yet connected).

## Server and routes

| Capability | Current behavior | Disposition | Now | Notes |
|---|---|---|---|---|
| Web server | Standard-library threaded HTTP server, one thread per request, no request timeout | redesigned | live | Planned: a single-worker typed API with one bounded fitting job at a time |
| Public pages and health | GET and HEAD for `/`, `/index.html`, `/healthz`, `/health` | retained | live | Health stays cheap; fitting must not block it |
| Sample list and download | `/api/samples`, `/api/sample?name=` for four bundled samples | retained | live | Samples are unchanged |
| Profile | `POST /api/profile`: column summary and a suggested target | redesigned | live, service | The service returns the same suggestion marked "suggested, needs confirmation", plus class counts, suggested exclusions and limit status |
| Analyze | `POST /api/analyze`: split, resample, compare three models | redesigned | live, service | Replaced by a versioned job API. The service call `train_for_app` runs the package comparison |
| Score new rows | `POST /api/score`: trains on all labelled rows, flags new rows at 0.5 | pending | live | To be kept separate from evaluation and to record whether the final model was refit |
| Convert | `POST /api/convert`: decode, unpack and convert an upload to CSV text | retained | live, service | The service reuses the same converters for uploads |
| Owner statistics and owner cookie | `/stats`, `/whoami/<key>` with an owner-only key | retained | live | Access path to be reviewed so the key never appears in logs |
| Request size limits | 45 MB request body, 30 MB decoded upload | redesigned | live, service | The service checks bytes before and after conversion (default 5 MB) and the package checks rows, columns, categorical cardinality and expanded width |

## Input formats

| Capability | Current behavior | Disposition | Now | Notes |
|---|---|---|---|---|
| CSV, TSV, semicolon, pipe, whitespace text | Delimiter, header, encoding, number-format and missing-value detection | retained | live, service | All 41 inputs of the legacy torture test are exercised through the service. Parsing notes and warnings are copied into the report |
| ARFF and KEEL | Parsed to a table | retained | live, service | |
| Excel | First non-empty sheet only | redesigned | live, service | Explicit sheet choice is planned |
| JSON and JSON Lines | Converted to a table | retained | live, service | |
| Stata | `.dta` read | retained | live, service | |
| ZIP and GZIP | Picks a member named like "train", otherwise the largest; GZIP output capped at 80 MB | redesigned | live, service | Explicit member choice and decompression bounds are planned |
| Refused formats | Parquet, Feather, SPSS, R, HDF5, SQLite, XML, PDF, Word give a clear refusal | retained | live, service | The service raises `APP_FORMAT_UNSUPPORTED` |
| Malformed rows | The legacy reader skips them and warns | redesigned | live, service | The service reports the skipped-row count in the report. The package's own CSV parser rejects them |

## Targets, labels and splits

| Capability | Current behavior | Disposition | Now | Notes |
|---|---|---|---|---|
| Target choice | Suggested column used unless the user changes it | redesigned | service | The caller confirms the target |
| Positive class | Rarest label, chosen silently | redesigned | service | Explicit and required; the rarest label is suggested |
| Multiclass target | Rarest label against all others, with a warning | redesigned | service | Explicit one-versus-rest |
| Label cleanup | Merges case and spacing variants, drops one-row classes and rows with a missing target | redesigned | service | Exact labels; one-row classes and missing targets are errors unless a reported policy is chosen |
| Identifier-like columns | Dropped silently when more than half of the values are distinct | redesigned | service | Suggested exclusions the caller confirms; never dropped silently |
| Train and test split | One stratified 25 percent split before resampling | redesigned | service | About 60 / 20 / 20 with validation, plus group, exact-feature group and chronological modes |
| Fold averaging | 3, 20 or 100 folds on uploads, with row caps | pending | live | The package does not repeat folds; bounded presets are to be decided from measurements |
| Decision threshold | Fixed 0.5 | redesigned | service | Selected on validation data |
| Threshold slider | Moves the cut on held-out probabilities | pending | live | To remain an exploratory view that never replaces the validation-selected result |

## Models, resampling and preprocessing

| Capability | Current behavior | Disposition | Now | Notes |
|---|---|---|---|---|
| Model comparison | Plain forest, CSRBoost and ERF-CSRBoost on the same held-out rows | redesigned | service, package | A stable registry of 12 recipes with identifiers, settings and seeds; defaults are the logistic, weighted forest, balanced forest, gradient boosting and ERF recipes |
| Plain forest | Weighted, 200 trees | redesigned | package | Replaced by an unweighted 100-tree forest and a separately named weighted forest. The old recipe is named, not runnable |
| CSRBoost | Re-implementation, full-depth trees, silent fallback | redesigned | service, package | Strict failure instead of a fallback; see the recipe page |
| ERF-CSRBoost | 20 bags of 50 trees, not boosting | retained | service, package | Same bag and tree counts and seed rule; differences are listed |
| Resampling methods | Nine methods, errors fall back to random oversampling | pending | live | SMOTE and clustered SMOTE map to package recipes. Random over- and undersampling, Borderline-SMOTE, ADASYN, SMOTE plus Tomek and cluster centroids are "legacy only" until a decision is made |
| Classifiers for export and scoring | Logistic regression, random forest, AdaBoost | pending | live | Logistic regression and the forest map to recipes. AdaBoost alone is "legacy only" |
| Mixed-type preprocessing | Imputation, scaling, one-hot encoding | redesigned | service, package | A training-only adapter with categorical-aware resampling, unseen-category handling and width limits |
| Silent fallbacks | Resampler errors are replaced and noted in text | redesigned | service, package | Strict failure by default, with a stable reason code per model |
| Training caps | 4,000 rows for comparison | redesigned | service | Table limits and explicit refusal instead of sampling. Hosted limits are not yet measured |
| CPU use | `n_jobs=-1` in forests | redesigned | package | One thread per worker, set before imports |

## Diagnostics, outputs and records

| Capability | Current behavior | Disposition | Now | Notes |
|---|---|---|---|---|
| Leakage demonstration | Leaky versus honest resampling with a 120-tree forest | pending | live | To remain a labelled controlled demonstration on a common untouched population |
| Duplicate and proxy scan | Exact duplicate rows; single-column proxies by an in-sample score | redesigned | live, service | The package profile reports duplicate feature groups and conflicting patterns as diagnostics. An in-sample proxy score is not offered as independent evidence |
| Downloads | Balanced training CSV, untouched test CSV, honesty receipt, scored CSV, pipeline snippet | pending | live | To become explicit user-owned downloads with row-level data described separately |
| Reports | Page results only | redesigned | service, package | Versioned JSON (no NaN or Infinity) and standalone HTML with escaped labels |
| Usage records | SQLite file with event type, day, short detail, hashed visitor tag, owner flag | retained | live | Existing records and semantics are kept. Client address handling is to be reviewed |
| Tests | Backend unit tests, a torture script and two Node checks | retained | live | Kept as they are; the service has its own offline suite |

## Added beyond the live application

| Capability | Now | Notes |
|---|---|---|
| Saved-prediction evaluation (labels, scores, thresholds, review budgets, hypothetical costs) | service, package | `evaluate_saved_predictions` wraps `rare_event_eval.evaluate_predictions`. The static browser workbench in the package stays offline and needs no upload |
| Typed errors with stable codes | service | Codes such as `APP_FORMAT_UNSUPPORTED`, `TABLE_CARDINALITY_EXCEEDED` and `LABEL_MISSING_TARGET`; messages hold no paths or tracebacks |
| Legacy name to recipe map | service | `legacy_recipe_map` lists each legacy classifier, resampling method and model with its recipe or "legacy only" and the reason |
