# HTTP API

This describes the service in `backend/api.py`: the pages (the workbench at `/` and the old page at `/classic`),
the `/api/v1` interface, the health routes and the legacy routes that the old page uses. Run it from the repository root with `python backend/api.py`. The `PORT`
variable sets the port (default 7860). The service listens on every network interface. Nothing in this file
says that a copy of it is deployed anywhere.

The service runs with one Uvicorn worker. Everything that reads or fits data runs in a child process that a
job coordinator starts, so a long fit does not block the web process (see `docs/ARCHITECTURE.md`, section
Service). Limits are provisional and every one can be changed by an environment variable (see
`docs/RESOURCE_LIMITS.md`).

## Conventions

- Bodies are JSON unless a route says otherwise. JSON the service writes is strict (no `NaN`, no `Infinity`) and
  ASCII only. The characters `<`, `>` and `&` are written as the six-character JSON escapes `\u003c`, `\u003e`
  and `\u0026`, which a JSON parser reads back as the same text.
- Every response has an `X-Request-ID` header, `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`
  and `X-Frame-Options: DENY`. Every response except the old page (`/classic`, which keeps the headers it always had) also sends
  `Referrer-Policy: no-referrer`. There is no CORS header on any route.
- A request with an `Origin` header whose host differs from the `Host` header is refused with 403
  `ORIGIN_NOT_ALLOWED` when its method is POST, PUT, PATCH or DELETE. A request with no `Origin` header (curl, a
  server) is not affected. This is a same-origin guard for browsers, not authentication. The comparison is of
  host and port only: the scheme is not compared (an `Origin` of `https://host:port` passes for a service that
  is reached as `http://host:port`), and a page that controls both the `Host` and the `Origin` the browser
  sends, as in DNS rebinding, passes it. The `/api/v1` capabilities still protect the versioned routes; the four
  legacy POST routes carry no capability, as on the old server.
- Sizes are enforced while the body arrives, so a chunked upload with no `Content-Length` header is stopped as
  well: 8 MB for `POST /api/v1/datasets` (provisional, `FYD_UPLOAD_LIMIT_BYTES`), 128 KiB for
  `POST /api/v1/jobs`, 45 MB for the four legacy POST routes, 64 KiB for anything else. A client that is still
  sending when the answer goes out may see the connection close before it has read the answer.

## Authentication: capabilities

There are no accounts. Creating a dataset returns a dataset capability, creating a job returns a job
capability. Each is a random URL-safe string of 43 characters, shown once, in the response body only. The
service keeps only a SHA-256 digest and compares in constant time. Send a capability as

```
Authorization: Bearer <capability>
```

and never in a URL. A dataset capability is used for `POST /api/v1/jobs` and `DELETE /api/v1/datasets/{id}`. A
job capability is used for every `/api/v1/jobs/{id}` route. A missing, malformed, wrong or foreign capability
on an existing id gets exactly the same 404 `NOT_FOUND` as an id that does not exist, so a caller cannot find
out which ids exist. The service never writes a capability to a log line, a report or an error body. Anyone
who holds a capability can use it, so treat it like a password. After a restart no capability is valid.

## Errors

Every error from `/api/v1` has this shape and the `X-Request-ID` header of the same request:

```json
{"error": {"code": "QUEUE_FULL", "message": "The service is busy with another job. Try again shortly.",
           "details": {"retry_after_seconds": 15}, "request_id": "9f1c0d2a7b3e4c58"}}
```

Messages and details are fixed texts or short values chosen by the service. They hold no traceback, no local
path, no capability and no row of the uploaded table. A response with status 429 or 503 that has
`details.retry_after_seconds` also has a `Retry-After` header.

| Status | Code | Meaning |
|---|---|---|
| 400 | `INVALID_JSON` | The body is not valid UTF-8 JSON (the constants `NaN` and `Infinity` are refused) |
| 400 | `INVALID_CONTENT_LENGTH` | The `Content-Length` header is not a number |
| 400 | `UPLOAD_EMPTY`, `UPLOAD_INVALID` | The upload is empty or not bytes |
| 403 | `ORIGIN_NOT_ALLOWED` | Cross-origin state-changing request |
| 404 | `NOT_FOUND` | Unknown route, unknown id, or a capability that does not belong to the id |
| 404 | `EXPORT_UNKNOWN` | The export kind is not on the allowlist |
| 405 | `METHOD_NOT_ALLOWED` | The route exists but not for that method (an `Allow` header is sent) |
| 409 | `RESULT_NOT_AVAILABLE` | The job ended without a report (see Result below) |
| 410 | `JOB_EXPIRED`, `DATASET_EXPIRED` | The files were deleted after their lifetime; only the right capability sees this |
| 413 | `REQUEST_TOO_LARGE`, `UPLOAD_TOO_LARGE` | Body larger than the limit for the route (`details.limit_bytes`) |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Wrong `Content-Type` (`details.allowed`) |
| 422 | `REQUEST_INVALID` | Unknown field, wrong type or missing field; `details.errors` lists field and kind, `details.unknown_fields` names unknown fields. Also `TOO_MANY_MODELS` |
| 422 | package and app codes such as `APP_INPUT_UNREADABLE` | The upload could not be read or profiled; the message is the package's, with paths removed |
| 422 | `MEMORY_LIMIT_EXCEEDED`, `RESULT_TOO_LARGE` | The worker went past its memory ceiling or produced a result over the size cap. For a job these are a `failed` state with that `reason.code`; the legacy routes answer 400 in their own shape |
| 429 | `QUEUE_FULL` | The active slot and the waiting slot are taken |
| 429 | `DATASET_BUSY` | The dataset already has a job that has not finished |
| 429 | `DATASET_LIMIT_REACHED` | Too many datasets are stored |
| 500 | `INTERNAL_ERROR`, `JOB_CRASHED`, `JOB_INTERNAL_ERROR` | A fault in the service or a worker; only the class of the fault is shown |
| 500 | `EXPORT_FAILED`, `EXPORT_TOO_LARGE` | The report could not be rendered, or the bundle would be larger than its bound |
| 503 | `STORAGE_FULL`, `SERVICE_UNAVAILABLE`, `LIGHT_TIMEOUT`, `JOB_TIMEOUT` | The service cannot take the work now |

Whether a model could be fitted is not an HTTP error. A job that was accepted ends in a state, and the
failure reason is in the job status and in the report.

## Routes

### Pages

| Route | Behaviour |
|---|---|
| `GET` and `HEAD` `/`, `/index.html` | The workbench: `index.html` of the folder `evaluation/src/rare_event_eval/web` (resolved from the repository root, or the folder named by `FYD_WEB_DIR`). `Cache-Control: no-store`. A `GET` writes a usage event of kind `visit`, exactly as the old page did; `HEAD` does not. |
| `GET` and `HEAD` `/<file>` | Every other regular file directly inside that folder, at its own root path (`/app.js`, `/style.css`, `/recorded-example.json`, ...). |
| `GET` and `HEAD` `/classic` | The old single page, the bytes of `frontend/index.html`, with the visit logging and headers `/` had before the workbench. Integrated replacements for resampling, Leakage Lab and scoring are in the new interface; the legacy receipt and extra methods remain here. |

How the files are chosen. The list is built once at startup from the folder, and the URL can only pick a name from
that list: no request text is ever joined into a path. A file is on the list when it is a regular file (not a link),
its name is plain (letters, digits, `.`, `-`, `_`, not starting with a dot, at most 100 characters), its extension is in
a fixed table and it is at most 8 MB. The table: `.html` `text/html; charset=utf-8`, `.css` `text/css; charset=utf-8`,
`.js` `text/javascript; charset=utf-8`, `.json` `application/json`, `.png` `image/png`, `.svg` `image/svg+xml`, `.ico`
`image/x-icon`, `.txt` `text/plain; charset=utf-8`. Everything else answers the same `404 not found` text as any unknown
path: hidden files, sub-folders and the files in them, other extensions, other letter cases, a trailing slash, encoded
dots and slashes, `..`. A file added after startup is not served; a file removed after startup is a 404.

Headers. Pages are `no-store`, other files are `public, max-age=300`, all carry `X-Content-Type-Options: nosniff`.
Any page other than `index.html` (for example the recorded example report) and every SVG also carries
`Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action
'none'; sandbox`. `index.html` (`/` and `/index.html`, GET and HEAD) carries `Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`: the workbench uses no inline script, style attribute or event handler, its downloads are `blob:` anchors, which these directives do not govern, and its icon is a `data:` URL. Other assets and `/classic` carry no policy (the old page uses inline script). `HEAD` returns the headers of `GET`
(including `Content-Length`) and no body. If the folder has no `index.html`, `/` and `/index.html` answer 503 with a
short text and the other files, `/classic` and `/api/v1` keep working. The page routes need no capability and take no
query parameter.

### Health and information

| Route | Behaviour |
|---|---|
| `GET /healthz`, `HEAD /healthz`, `GET /health`, `HEAD /health` | Plain text `ok`, status 200. These handlers take no lock and read no file, so they answer while a fit is running. |
| `GET /api/v1/version` | `{"app_version", "package_version", "report_schemas", "release_id"}`. `release_id` comes from `FYD_RELEASE_ID` if it looks like an identifier (letters, digits, `.`, `_`, `-`, up to 64 characters), else `unknown`. No path and no secret. |
| `GET /api/v1/samples` | The bundled sample datasets, same content as `GET /api/samples`. |

### Datasets

`POST /api/v1/datasets` uploads one table as the raw request body.

- `Content-Type` must be `application/octet-stream` or `text/csv` (parameters such as `charset` are ignored),
  otherwise 415.
- `X-Filename` (optional) names the file, so the reader can pick the format (`.xlsx`, `.json` and so on).
- The size is limited while streaming (413). The upload is stored in the job storage, profiled in a worker
  process and answered with status 201:

```json
{"dataset_id": "ds_...", "capability": "<shown once>", "expires_at": "2026-11-09T12:30:00Z",
 "limits": {"upload_limit_bytes": 8388608, "max_rows": 10000, "max_features": 100, "max_models_per_job": 5,
            "per_model_timeout_seconds": 60.0, "job_timeout_seconds": 300.0, "queue_wait_seconds": 120.0,
            "dataset_idle_seconds": 1800.0, "finished_job_seconds": 1800.0},
 "profile": {"schema": "app-profile-1.0", "rows": 300, "target": {"suggestion": {"column": "label"}},
             "classes": {"suggested_positive": {"label": "fraud", "status": "suggested, needs confirmation"}}}}
```

The profile suggests a target and a positive label but never applies them: the caller names both when it
creates a job. A file that cannot be read is a 422 with the package's code and leaves nothing stored.
`DELETE /api/v1/datasets/{id}` (dataset capability) cancels any live job on the dataset, removes the files and
answers `{"deleted": true}`.

A dataset is deleted after `dataset_idle_seconds` (30 minutes) without use, but never while one of its jobs is
queued or running.

### Jobs

`POST /api/v1/jobs` with `Content-Type: application/json` and the dataset capability in the bearer header.

```json
{"dataset_id": "ds_...", "target": "label", "positive_label": "fraud", "negative_label": "ok",
 "models": ["logistic", "rf_weighted"], "seed": 7}
```

| Field | Type | Notes |
|---|---|---|
| `dataset_id` | text | required |
| `target` | text | required, a column name |
| `positive_label` | text or number | required; never a boolean, null or container |
| `negative_label` | text or number | give it for a binary target |
| `one_vs_rest` | boolean | `true` instead of `negative_label` for more than two labels |
| `models` | list of model ids | at most 5 (provisional); default is the package's default set |
| `split_mode`, `calibration`, `missing_policy` | text | passed to the package, which checks the value |
| `seed` | integer | |
| `id_columns`, `exclude_columns` | lists of column names | columns left out of the predictors |
| `group_column`, `time_column` | column names | for group and time splits |
| `review_fractions` | list of up to 10 numbers | |
| `costs` | object | `false_positive_cost`, `false_negative_cost`, optional `review_cost` |

Any other field is a 422 that lists it in `details.unknown_fields`. Whether a model id or a column exists is
decided by the package: a wrong one makes the job end `failed` with the package's code, not an HTTP error.

The answer is `202` with `Location` and

```json
{"job_id": "job_...", "capability": "<shown once>", "state": "queued", "poll_url": "/api/v1/jobs/job_..."}
```

It is `429` when the service has no room (`QUEUE_FULL`) or when the dataset already has a live job
(`DATASET_BUSY`), immediately and with `Retry-After`.

`GET /api/v1/jobs/{id}` returns the state:

```json
{"job_id": "job_...", "state": "running", "queue_position": null, "elapsed_seconds": 4.2,
 "limits": {"per_model_timeout_seconds": 60.0, "job_timeout_seconds": 300.0, "queue_wait_seconds": 120.0},
 "progress": {"stage": "fitting", "models_selected": ["logistic", "rf_weighted"],
              "models_completed": [{"model": "logistic", "status": "succeeded"}],
              "current_model": "rf_weighted", "limits": {}, "warnings": [], "elapsed_seconds": 4.2},
 "models": [{"model": "logistic", "status": "succeeded"}], "warnings": [], "reason": null,
 "expires_at": null}
```

`stage` is one of `parsing`, `profiling`, `splitting`, `fitting`, `evaluating`, `reporting`, `done`. Progress is
counted events only (which models finished, which one is being fitted). The service never reports a percentage
or an estimated time, because it has no measurement to base one on. `reason` holds a `code` and a short
`message` when the job did not succeed.

`DELETE /api/v1/jobs/{id}` cancels the job and returns the same status document. It is idempotent: on a job
that has already ended it returns the current state and changes nothing. A queued job is removed. For a running
job the call returns after the whole worker process tree has been killed and reaped, and the state is then
`cancelled`.

### State machine

```
queued ---> running ---> succeeded | partial | failed | cancelled | timed_out | interrupted
queued ---> cancelled | timed_out | interrupted
(any ended state) ---> expired        (by the lifetime sweep only)
```

- `succeeded`: every selected model produced a result.
- `partial`: at least one model produced a result and at least one failed, timed out or was cancelled. The
  report lists the failed models with reason codes and never drops them.
- `failed`: no model produced a result, or the request as a whole failed (for example a column that is not in the
  file).
- `timed_out`: the whole job exceeded `job_timeout_seconds` (`JOB_TIMEOUT`), it waited longer than
  `queue_wait_seconds` in the queue (`QUEUE_WAIT_EXCEEDED`), or every model reached the per-model limit
  (`MODEL_TIMEOUT`).
- `interrupted`: the service was stopped while the job was queued or running.
- `expired`: the files were deleted after `finished_job_seconds` (30 minutes). The record stays for 24 hours so
  the right capability gets a 410.

After a restart no job survives: the service deletes its whole storage folder on start, so jobs and datasets of
the previous process are unknown and every request for them is a 404.

### Result

`GET /api/v1/jobs/{id}/result`

| Job state | Status | Body |
|---|---|---|
| `succeeded`, `partial` | 200 | the `training-1.1` report of the evaluation package (see `evaluation/docs/report-contract.md`) |
| `queued`, `running` | 202 | the status document, with `Retry-After: 2` |
| `failed`, `cancelled`, `timed_out`, `interrupted` | 409 | error `RESULT_NOT_AVAILABLE`, `details` holds `state`, `reason` and the per-model list |
| `expired` | 410 | error `JOB_EXPIRED` |

### Exports

`GET /api/v1/jobs/{id}/exports/{kind}` with the job capability, for a job in state `succeeded` or `partial`
(409 and 410 as above). The allowlist is closed; any other `kind` is a 404 `EXPORT_UNKNOWN`.

| Kind | Content type | File name | Content |
|---|---|---|---|
| `report_json` | `application/json` | `fyd-report.json` | the same report as `/result` |
| `report_html` | `text/html; charset=utf-8` | `fyd-report.html` | the package's standalone report page |
| `config_json` | `application/json` | `fyd-config.json` | the settings of the run (the request, every default written out), the limits and the input hash |
| `repro_bundle` | `application/zip` | `fyd-repro-bundle.zip` | a zip with the config, the report as JSON and HTML, the versions, a README and the command that repeats the run |

All exports are sent with `Content-Disposition: attachment`, `Cache-Control: no-store` and `nosniff`. None of them
holds a row of the uploaded table, a prediction or score per row, the row indexes of the split (the report has only
counts and a hash of each index list) or a fitted model.

Row-level exports are not part of this release. Row mappings, split indexes and per-row scores are a deliberate later
item: if they are added they will be separate kinds, behind an explicit user action, with their privacy implications
explained, and they will not appear in the default aggregate exports. There is no export kind for them now.

**Provenance label.** The report that `/result` and every export return carries two fields added by the service, on
top of the package's `training-1.1` fields: `"provenance": "fresh app run"` and an `app` block (`app_version`,
`release_id`, a note). The four labels used across the project are `fictional fixture`, `recorded app run`,
`fresh app run` and `historical research record`. Existing `training-1.1` consumers ignore the added fields. The
HTML export prints the label and what it means in its header. A result produced by this app is never presented as a
historical ERF research result.

**`config_json`.** `schema` stays `fyd-job-config-1.0`; the fields `provenance`, `effective` and `app` were added.
`request` is the validated request as sent. `effective` is the same request with every default written out (models,
split mode, seed, calibration, missing-target policy, review fractions, costs, excluded and identifier columns,
group and time columns, the per-model time limit and the table limits the job ran with). The defaults are read from
`services.train_for_app` itself, so they cannot drift from what the job used.

**`repro_bundle`.** A zip built in memory, at most 40 MB uncompressed and 20 MB compressed (a larger one is a 500
`EXPORT_TOO_LARGE`). The member order is fixed, all timestamps are 1980-01-01 00:00:00, attributes are fixed and the
compression level is fixed, so the same job gives the same bytes on one Python and zlib build.

| File | Content |
|---|---|
| `config.json` | the `config_json` document plus `bundle_schema: fyd-repro-bundle-1.0` |
| `report.json` | byte for byte the `report_json` export |
| `report.html` | byte for byte the `report_html` export |
| `versions.json` | app version, source release id (`FYD_RELEASE_ID`, else `unknown`), package version, report and config schema ids, Python, numpy, scikit-learn, imbalanced-learn and pandas versions, and the recipe id, version, scope and family of every model |
| `README.txt` | what the bundle is, what it does not contain (the table, any row, any per-row prediction, the row indexes, any model), a privacy note, and that the result is not a historical ERF result |
| `replay.txt` | the exact command, the notes below and the same arguments as a JSON list |

**The replay command.** `replay.txt` holds one `python -m rare_event_eval train ...` command, written for a POSIX shell
and for Windows PowerShell (the PowerShell form is left out when a value holds a double quote, which Windows PowerShell
5.1 cannot pass safely; the JSON argument list is always exact). It uses only flags the command line has. Every value
option is written as `--name=value`, so a label that begins with a dash cannot be read as an option. It sets the models,
the split mode, the group or time column, the seed, the calibration, the positive and negative label or
`--one-vs-rest`, the target, the excluded and identifier columns, the review fractions, the error and review costs, the
missing-target policy, the per-model time limit and `--max-expanded-features` (which also makes the command line use the
mixed-type adapter, the code path the hosted job runs). The data file is named `YOUR_INPUT_FILE.csv`: the bundle does
not hold the data, and `replay.txt` gives the SHA-256 and byte count of the file to use.

The command uses the `python -m rare_event_eval train` flags as they exist in the package command line (`cli.py`); that is verified, not assumed: the test parses the command with the package's own argument parser and then runs it.

This is tested. `tests/test_exports.py` runs a job, reads the command from the bundle, runs it on the original file and
compares the result with the hosted report: metrics, thresholds, review-budget results, calibration bins, class counts,
split counts and index hashes, label mapping, predictors, preprocessing records and model records are equal. Two
scenarios are covered: a plain job with sigmoid calibration, costs and an excluded column, and a job with hostile
names, `one_vs_rest` labels, a group split, an identifier column and the `drop_reported` missing-target policy. The
fields that legitimately differ are exactly these, and the test fails on any other difference:

- `created_utc` and the timings and memory (`fit_seconds`, `fit_and_evaluation_seconds`, `worker_wall_seconds`,
  `peak_process_rss_bytes`);
- the whole `input_parsing` record: the app reads uploads with its own table reader and the command line uses the
  package reader (for a plain UTF-8 comma-separated file with a header both give the same table);
- the table limits echoed back (`max_rows`, `max_bytes`): the command line has no option for them and uses the package
  defaults (20,000 rows, 5,000,000 bytes), so a file above 5,000,000 bytes is refused by the command line;
- the `provenance` and `app` fields, which the app adds and the command line does not.

What the command cannot express is stated in `replay.txt` for the run it describes, not hidden: an upload the app
converted before reading (for example a spreadsheet), a reader warning (the app's reader drops a first column that is
only row numbers and skips malformed rows, the command line keeps the column and rejects the row), a file over the byte
limit of the command line, models that failed or timed out (a timeout depends on the speed of the machine), and a split
mode or calibration the command line does not offer. A value with a control character cannot be written on one line, so
only the JSON argument list is given.

How user text is treated. Labels and column names come from the uploaded table and can be hostile. In
`report_html` the package escapes every value it prints (a label `<script>alert(1)</script>` is printed as
`&lt;script&gt;alert(1)&lt;/script&gt;`, and quotes become `&#x27;`), the page has no script and no remote
asset, and it carries a content security policy (also sent as a header together with `sandbox`). A label that
looks like a spreadsheet formula, such as `=cmd|' /C calc'!A0`, is printed as text there. The JSON exports and
`/result` keep the label as a JSON string, so a client that parses the JSON gets the original text back, but
the bytes on the wire never contain `<`, `>` or `&`: they are written as the escapes `\u003c`, `\u003e` and
`\u0026`, six characters each. No export
is a CSV file, so there is no spreadsheet-formula path; a client that writes labels into a CSV or a
spreadsheet itself has to neutralise a leading `=`, `+`, `-` or `@`, which this service does not do for it. Bidirectional control characters (U+202A to U+202E, U+2066 to U+2069, U+200E, U+200F, U+061C) in labels, column names and other user text are printed in the HTML report as visible markers such as `<U+202E>` (escaped, shown as text), because they would otherwise reorder the visible text around them; the JSON exports keep the real characters. The HTML report also names the real source of the dataset description: a recorded example says the table was generated by a seeded script, a fresh run keeps "User-supplied".

## Curl walk-through

```bash
BASE=http://localhost:7860

# 1. Upload and profile (the response holds the dataset capability, shown once)
curl -s -X POST "$BASE/api/v1/datasets" -H "Content-Type: text/csv" -H "X-Filename: loans.csv" \
     --data-binary @loans.csv > dataset.json
DS_ID=$(python -c "import json; print(json.load(open('dataset.json'))['dataset_id'])")
DS_CAP=$(python -c "import json; print(json.load(open('dataset.json'))['capability'])")

# 2. Start a job
curl -s -X POST "$BASE/api/v1/jobs" -H "Content-Type: application/json" -H "Authorization: Bearer $DS_CAP" \
     -d "{\"dataset_id\": \"$DS_ID\", \"target\": \"label\", \"positive_label\": \"fraud\", \"negative_label\": \"ok\",
          \"models\": [\"logistic\", \"rf_weighted\"], \"seed\": 7}" > job.json
JOB_ID=$(python -c "import json; print(json.load(open('job.json'))['job_id'])")
JOB_CAP=$(python -c "import json; print(json.load(open('job.json'))['capability'])")

# 3. Poll the state, then fetch the report and the HTML page
curl -s "$BASE/api/v1/jobs/$JOB_ID" -H "Authorization: Bearer $JOB_CAP"
curl -s "$BASE/api/v1/jobs/$JOB_ID/result" -H "Authorization: Bearer $JOB_CAP" -o report.json
curl -s "$BASE/api/v1/jobs/$JOB_ID/exports/report_html" -H "Authorization: Bearer $JOB_CAP" -o report.html

# 4. Cancel a job that is still queued or running (safe to repeat)
curl -s -X DELETE "$BASE/api/v1/jobs/$JOB_ID" -H "Authorization: Bearer $JOB_CAP"

# 5. Remove the dataset
curl -s -X DELETE "$BASE/api/v1/datasets/$DS_ID" -H "Authorization: Bearer $DS_CAP"
```

A script that polls should wait at least a second between calls, stop on any state other than `queued` or
`running`, and honour `Retry-After`.

## Legacy routes

These keep the request and response shapes of `backend/server.py` so the old page (`/classic`) keeps working. The
workbench uses `/api/v1` and, for the bundled samples, `/api/samples` and `/api/sample`. They carry
no capability and are not part of the versioned interface.

| Route | Behaviour |
|---|---|
| `GET /classic` | the old page (`frontend/index.html`, the page `/` served before the workbench); a usage event of kind `visit` is written for GET (not HEAD) |
| `GET /api/samples`, `GET /api/sample?name=` | sample list and the raw CSV (`text/csv`) |
| `POST /api/profile`, `POST /api/convert` | JSON body as before; run in the light lane (one at a time, two waiting, then 429) |
| `POST /api/analyze`, `POST /api/score` | JSON body as before; run in the heavy queue (shared with `/api/v1` jobs); the usage events `analyze` and `score` are written before the run |
| `GET /stats`, `GET /stats/<key>`, `GET /whoami/<key>` | owner pages |

- The `HEAD` method is answered like GET for the pages and the owner pages.
- A cross-origin POST to one of the four legacy POST routes is refused with 403 in the legacy error shape.
- Errors are `400 {"ok": false, "error": "<text>"}`, as before, with the engine's own message. An unknown path
  or an unsupported method is `404` with the text `not found`. A busy service is `429` in the same shape with
  a `Retry-After` header. A body over 45 MB (or an invalid `Content-Length`) is a `400` in that shape, as
  before, not a 413.
- The result of the legacy engine is relayed as the engine produced it. Large uploads are written to the job
  storage first, so with the default storage cap a legacy request whose JSON body does not fit into the free
  space is refused with 503 `STORAGE_FULL` in the legacy shape.
- Result size. The legacy engine's analysis result carries the balanced training table, so it is about 2.6
  times the size of the input (measured with the engine on synthetic tables of 8 numeric columns, one category
  and a label: a 0.6 MB CSV gave a 1.7 MB result, a 2.6 MB CSV 6.7 MB, a 3.9 MB CSV 10.1 MB and a 7.8 MB CSV
  20.3 MB). The four legacy commands therefore have their own result cap, `FYD_MAX_LEGACY_RESULT_BYTES`,
  default 32 MB, separate from the 8 MB cap of the `/api/v1` jobs (`FYD_MAX_RESULT_BYTES`). A legacy result
  over its cap is refused with `400 {"ok": false, "error": "The result is larger than the service allows. The
  limit is 32 MB. Try a smaller file."}`. Remaining differences from the old server, which had no such cap and
  accepted a request body of up to 45 MB: a legacy analysis whose result would pass 32 MB (an input of roughly
  12 MB of CSV or more) is refused; a legacy request is also held to the 64 MB storage budget (the request file
  plus the 32 MB the job reserves for its result must fit beside everything else stored, so a request above
  roughly 32 MB, or a smaller one while datasets are stored or another legacy job is live, can get 503
  `STORAGE_FULL`); and every worker has a memory ceiling (next item), which with its default is the tighter
  limit for a legacy analysis.
- Memory ceiling. Every worker process tree (a v1 job, a legacy call, a profile or a conversion) is held to
  `FYD_CHILD_MEMORY_MB`, default 700, where 0 turns it off. A tree that goes past it is killed and the job
  fails with the code `MEMORY_LIMIT_EXCEEDED`; a legacy route answers 400 in its shape with the text "This
  request needs more memory than the service allows for one job. Try a smaller file." The ceiling is a
  provisional number that has not been measured on a hosted machine; it replaces the old server's behaviour of
  using memory until the machine ran out. Measured on a Windows development machine (peak committed memory of
  the worker tree, 10 columns): a legacy analysis of a 0.6 MB CSV peaked at 144 MB, 2.6 MB at 249 MB, 3.9 MB at
  379 MB and 7.8 MB at about 1 GB, so with the default ceiling a legacy analysis of a CSV above roughly 6 MB
  (more rows or more columns than that table counts for less) fails with `MEMORY_LIMIT_EXCEEDED`; a v1 job on a
  synthetic 30-column table peaked at 313 MB (300 rows), 362 MB (3,000 rows) and 480 MB (10,000 rows).
- When the client disconnects, or the request is cancelled, the job is cancelled and its process tree is
  killed.
- Owner statistics key: accepted as `X-Stats-Key`, `?key=` or in the path, compared in constant time, never
  logged by this service. Without a configured key (`STATS_KEY`) or with a wrong one the answer is 404. Prefer
  the header: a key in a URL ends up in browser history and in the logs of any proxy in front of the service.
- Visitor hash: the client address is only used for the daily pseudonymous tag in `track.py`. `Fly-Client-IP` is
  trusted only when the environment variable `FLY_APP_NAME` is set (Fly sets it in its machines, and its proxy
  overwrites that header) or when `FYD_TRUST_FLY_CLIENT_IP=1` is set; anywhere else any caller can send the
  header, so it is ignored. `X-Forwarded-For` is ignored unless `FYD_TRUST_X_FORWARDED_FOR=1`. With neither
  header trusted the socket peer is used. New referrers are reduced to hostnames; paths, queries and owner URLs are not stored.
  `FYD_USAGE_DB` can name another database file for the usage log.
- Usage statistics: see [USAGE-STATS.md](USAGE-STATS.md) for recognised source tags, UTC date filters,
  owner/test/bot exclusions, new workbench request counts and limits of historical attribution.
- No per-visitor resource quotas. The aggregate usage log does not enforce request, job or byte quotas
  per visitor or address. One visitor can occupy the single heavy slot and its waiting slot, the dataset slots and the
  light lane, and other visitors then get 429 until a slot is free. That is an availability limit of a small
  shared service and a known limit, not an ownership problem: the other visitors cannot see, cancel or delete
  anything that visitor created, and a running job of theirs is never touched.

## Honest limits

- One process, one active heavy job, one waiting. This is a small demo service, not a batch system.
  Linux/Windows and Python 3.11/3.12 checks passed, and the exact Linux deployment image was exercised
  at one CPU and 1,024 MiB. See [release verification](RELEASE-VERIFICATION.md) for the measured scope.
- All state is in memory and in one temporary folder. A restart loses every dataset and job. Two service
  processes must not share a storage folder.
- Capabilities are bearer secrets over whatever transport the service is exposed on. The service does not
  terminate TLS; on a hosted deployment that is the proxy's job.
- Timeout, row, feature and storage limits are bounded presets, not guarantees that every admissible input
  finishes. Constrained-image fixtures and small live jobs were checked; arbitrary maximum inputs were not.
  A worker has thread limits and a memory ceiling (`FYD_CHILD_MEMORY_MB`, 700 by default);
  an out-of-memory kill by the operating system of the whole machine still shows as `JOB_CRASHED`.
- The report is the package's report for the data that was uploaded. The service does not check the data for
  leakage, time order or label quality beyond what the profile and the package report.
- The HTML export is rendered in the web process by the package renderer, not in a worker.

## Integrated data tools

### Import table selection

Dataset uploads may send `X-Import-Selection`, a percent-encoded UTF-8 JSON object containing `member`
and/or `sheet`. If multiple supported members or worksheets exist, the service returns 422
`APP_IMPORT_SELECTION_REQUIRED` with `selection_kind` and `choices`. The new interface displays a
selector and resubmits only after a choice. The selection follows the dataset into training and its
parsing record. The input hash remains the original uploaded bytes. Names are matched in memory,
never extracted to disk. Unsafe paths, duplicate archive names, excessive expansion and nesting are
refused. GZIP expansion is bounded by the upload byte limit; ZIP/XML total expansion is at most
32 MB and at most four times that limit. Workbooks are read with a row bound. Legacy converters
remain compatible and are separate from the new import policy.

### Operations

`POST /api/v1/tools/{operation}`, JSON, where operation is `resample`, `leakage` or `score`.
Required: `csv`, `target`, exact `positive_label`, `negative_label`, `confirm_labels: true`.
Optional: `exclude_columns`, `seed`, `method`, `model`, `score_csv`, `include_row_data: true`.
UTF-8 CSV only; 2 MB per table, 2,000 rows, 50 predictors, 40 categories per predictor and 200 expanded
features. The aggregate request body is also bounded (at most 4 MB). Same-origin rules apply.

The synchronous caller response uses the bounded heavy lane and disconnect cancellation. It returns
aggregate settings/counts/warnings and an empty `downloads` mapping by default. Row files appear only
after explicit `include_row_data: true`. There is no public result id or download URL. A caller supplies
its own inputs; it cannot read an existing dataset or another caller's job through this endpoint.
Validation errors return a typed envelope; worker input failures currently use the coordinator's legacy
error translation. Unknown fields are ignored, not forwarded to the worker.

Resampling supports `none`, `random_over`, `random_under`, `smote` and `clustered_smote`. Only training
rows fit preprocessing/resampling; validation/test exports retain original cells. Exports include
partitions and source-pattern candidates, marking synthetic rows explicitly. Formula strings are
neutralized without changing numeric analytical values. Training values may be imputed.

Scoring supports `logistic`, `rf_raw` and `erf`, refits on all labelled rows and flags at 0.5. It does not
claim held-out performance or calibration. Existing input column names are preserved using collision-safe
score/flag names. `score_csv` must contain all selected predictors.

The Leakage Lab accepts SMOTE or clustered SMOTE. The honest sampler sees training rows only; the
deliberately incorrect sampler sees all rows to generate synthetic training additions. Both models are
tested on identical original test rows. Different training sizes and synthetic rows are confounders;
either arm can perform better. It is a teaching intervention, not a pipeline leakage detector.
