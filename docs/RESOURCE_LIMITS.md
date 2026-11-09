# Resource limits

These are the limits of the service in `backend/api.py` and `backend/jobs/`. Every value below is a provisional
preset chosen before measurement on a hosted machine. Local production-image verification at one CPU and
1 GB passed 44 checks with 13 memory samples peaking at 335.6 MiB for its small fictional tables. This is
not a maximum-size-input benchmark or a measurement of the Fly host. See `RELEASE-VERIFICATION.md`.

## Provisional presets

| Setting | Default | Environment variable | What happens at the limit |
|---|---|---|---|
| Heavy jobs running at once | 1 | `FYD_MAX_ACTIVE_HEAVY` | the next one waits |
| Heavy jobs waiting | 1 | `FYD_MAX_WAITING_HEAVY` | 429 `QUEUE_FULL` with `Retry-After`, at once |
| Live (queued or running) jobs per dataset | 1 | `FYD_MAX_JOBS_PER_DATASET` | 429 `DATASET_BUSY` |
| Datasets stored | 4 | `FYD_MAX_DATASETS` | 429 `DATASET_LIMIT_REACHED` |
| Upload size (raw body) | 8 MB | `FYD_UPLOAD_LIMIT_BYTES` | 413, checked while the body streams |
| Rows per table | 10 000 | `FYD_MAX_ROWS` | the profile reports the violation; a job on it fails with the package's code |
| Expanded predictor features | 100 | `FYD_MAX_FEATURES` | same (this is the package's `max_expanded_features`) |
| Models per job | 5 | `FYD_MAX_MODELS` | 422 `TOO_MANY_MODELS` |
| Time per model | 60 s (1 to 600) | `FYD_MODEL_TIMEOUT_SECONDS` | that model is reported `timeout`; the job is `partial`, or `timed_out` if every model timed out |
| Time per job | 300 s | `FYD_JOB_TIMEOUT_SECONDS` | the worker tree is killed, state `timed_out`, code `JOB_TIMEOUT` |
| Wait in the queue | 120 s | `FYD_QUEUE_WAIT_SECONDS` | state `timed_out`, code `QUEUE_WAIT_EXCEEDED` |
| Light lane (profile, convert) run time | 20 s, one at a time | `FYD_LIGHT_TIMEOUT_SECONDS` | the task is killed, 503 |
| Light lane waiting callers | 2 | `FYD_LIGHT_MAX_WAITING` | 429 at once |
| Dataset idle lifetime | 30 min | `FYD_DATASET_IDLE_SECONDS` | files deleted, the owner gets 410 |
| Finished job lifetime | 30 min | `FYD_FINISHED_JOB_SECONDS` | files deleted, state `expired` |
| Tombstone (so the 410 can be answered) | 24 h | `FYD_TOMBSTONE_SECONDS` | the id becomes a plain 404 |
| Stored bytes, all datasets and jobs | 64 MB | `FYD_MAX_STORED_BYTES` | new uploads refused with 503 `STORAGE_FULL`; the oldest finished job is removed if the folder grows past the cap |
| Result size, `/api/v1` jobs and profiles | 8 MB | `FYD_MAX_RESULT_BYTES` | the job fails with `RESULT_TOO_LARGE` |
| Result size, the four legacy commands | 32 MB | `FYD_MAX_LEGACY_RESULT_BYTES` | the call fails with `RESULT_TOO_LARGE`; a legacy route answers 400 in its shape |
| Memory of one worker process tree | 700 MB | `FYD_CHILD_MEMORY_MB` (0 turns it off) | the tree is killed, the job fails with `MEMORY_LIMIT_EXCEEDED` |
| Reproducibility bundle, built in the web process | 40 MB uncompressed, 20 MB compressed (fixed) | not configurable | 500 `EXPORT_TOO_LARGE`; the report it holds is already limited to `FYD_MAX_RESULT_BYTES` |
| One workbench file served from the web folder | 8 MB (fixed) | not configurable | the file is not on the allowlist and answers 404 |
| Job request JSON | 128 KiB body, 64 KiB of parameters | none (fixed in code) | 413 or 422 |
| Legacy request body | 45 MB | none (fixed in code) | 400 in the legacy shape |
| Concurrent connections | 64 | none (fixed in `main()`) | Uvicorn answers 503 |
| Keep-alive | 5 s | none (fixed in `main()`) | the connection is closed |
| Integrated data tools | 2 MB per CSV, 2,000 rows, 50 predictors, 40 categories and 200 expanded features | fixed | the worker refuses invalid input; the overall JSON body is capped at 4 MB |

Also fixed in `main()`: one worker, access log off.

## How the settings are read

- At start the service reads `FYD_<NAME>` for every field of `JobConfig` (the table lists them) and uses the
  default for a variable that is missing or empty. A value that does not parse as the right type is ignored,
  silently, and the default is used. A value that parses but is out of range (for example `FYD_MODEL_TIMEOUT_SECONDS=0`)
  stops the service at start with an error.
- Other variables: `PORT` (default 7860), `FYD_TEMP_ROOT` (the private storage folder, default
  `<system temp>/fyd-jobs`; the service deletes everything in it at start and refuses a non-empty folder that
  it did not create), `FYD_RELEASE_ID` (shown by `/api/v1/version` and written into reports and bundles), `FYD_WEB_DIR` (the folder of
  the workbench files; default `evaluation/src/rare_event_eval/web`), `FYD_TRUST_X_FORWARDED_FOR=1`,
  `FYD_TRUST_FLY_CLIENT_IP=1` (trust the `Fly-Client-IP` header outside Fly; on Fly the variable `FLY_APP_NAME`,
  which Fly sets, already does that, and without either the header is ignored), `FYD_USAGE_DB`, `STATS_KEY`,
  and for tests only `FYD_ALLOW_TEST_COMMANDS=1`.
- Workers are started with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`
  and `VECLIB_MAXIMUM_THREADS` set to 1 and `LOKY_MAX_CPU_COUNT=1`. Variables whose names contain KEY, SECRET,
  TOKEN, PASSWORD or CREDENTIAL are removed from a worker's environment.

Example: allow a longer fit and a bigger upload on a machine that has been measured for it.

```bash
FYD_JOB_TIMEOUT_SECONDS=600 FYD_MODEL_TIMEOUT_SECONDS=120 FYD_UPLOAD_LIMIT_BYTES=16777216 \
FYD_MAX_STORED_BYTES=268435456 python backend/api.py
```

## Memory ceiling of a worker

Every worker process tree gets the same ceiling, on every lane (heavy jobs, the light lane, legacy calls).

- Windows: the ceiling is the job-wide memory limit of the Job Object the tree already lives in, so the system
  refuses allocations beyond it for the whole tree (the root, the model workers, and anything they start). A
  completion port tells the supervisor, which kills the tree and fails the job with `MEMORY_LIMIT_EXCEEDED`.
  The limit counts committed memory, which is at least the resident memory.
- Linux and other POSIX systems: the supervisor adds up the resident memory (RSS from `/proc/<pid>/stat`) of the
  root, its descendants and every process still in its process group, at most every 0.5 s, with the standard
  library only. When the sum passes the ceiling it kills the tree and fails the job the same way. This is
  sampling: a program that allocates faster than half a second per ceiling can pass it before the next sample,
  and pages shared by forked processes are counted once per process. If the process table cannot be read, no
  ceiling is enforced.
- A ceiling of 0 turns it off. 700 MB is a provisional number chosen for a small hosted machine; it has not been
  measured there. Measured on a Windows development machine (peak committed memory of the whole worker tree,
  default model set): a v1 job on a synthetic 30-column mixed table peaked at 313 MB with 300 rows, 362 MB with
  3,000 rows and 480 MB with 10,000 rows (the row limit), so the default leaves about 1.4 times headroom at the
  limit. A legacy analysis peaked at 144 MB for a 0.6 MB CSV, 249 MB for 2.6 MB, 379 MB for 3.9 MB and about
  1 GB for 7.8 MB, so with the default ceiling the legacy analysis stops at a CSV of roughly 6 MB, well below
  what the 32 MB result cap alone would allow. Raise `FYD_CHILD_MEMORY_MB` on a machine that has the memory.
- A ZIP or spreadsheet that expands to a large size is bounded by this ceiling rather than by a smaller size
  check: the converter in the legacy engine accepts a ZIP member declared up to 200 MB, and the conversion runs
  inside the worker.

## Things that interact

- The storage cap counts the request file of a legacy call, and a legacy job also reserves its own result cap
  (32 MB by default, 4 MB for a profile, whose result holds statistics only) while it is live. New uploads, new
  jobs and new legacy calls are admitted only if what is stored, plus the reserves of live legacy jobs, plus the
  new request and its reserve, stays within `FYD_MAX_STORED_BYTES`. A legacy analyze or score call with a 30 MB
  file sent twice is a JSON body of about 80 MB, which is already over the 45 MB legacy body limit; a single
  30 MB upload becomes about 40 MB of base64 JSON, so with the default 64 MB cap and a 32 MB reserve it is
  refused with `STORAGE_FULL` (a request body above roughly 32 MB, or less while other data is stored or another
  legacy job is live). The old server had no such budget. Raise `FYD_MAX_STORED_BYTES` if the page's 30 MB
  uploads must keep working.
- A legacy result is read into the web process, parsed, and written out again, so a 32 MB result can briefly
  need roughly three times its size in the web process. This has not been measured.
- Result caps differ by lane: legacy results up to 32 MB are relayed; `/api/v1` results and dataset profiles
  keep the 8 MB cap. The old server had no result cap at all, so a legacy analysis whose result would pass
  32 MB (an input of roughly 12 MB of CSV or more) is a remaining difference.
- No per-visitor accounting in this release. One visitor can hold the heavy slot, the waiting slot, the dataset
  slots and the light lane, and others then receive 429. This is a known availability limit of a small shared
  service, not an ownership problem: capabilities keep every visitor's datasets and jobs private.
- A legacy request body is held in memory in the web process while it is parsed and written to disk (the body,
  the parsed text and the re-encoded file can exist at the same time, roughly three times its size). The
  machine's memory has to allow for that at the largest legacy body, and this has not been measured.
- The queue wait limit (120 s) is shorter than the job limit (300 s), and a caller can wait for one running job
  plus its own run, so a client timeout below roughly 7 minutes can end before the service gives up.
- `max_models_per_job` times the per-model timeout is larger than the job timeout at the defaults (5 x 60 s against
  300 s), so a job that fills all five models with slow ones is stopped by the job limit, not the model limit.

## Not yet measured

- Memory of a worker for tables at the row and feature limits, and of the web process with a worker running.
  The worker ceiling above is a guess until it is measured on the hosted machine type.
- Fit times of the default model set at 10 000 rows and 100 features on the hosted machine type; the time
  limits above are guesses until they are timed there.
- Behaviour under the hosted proxy (connection limits, idle timeouts, request buffering, the client address
  header), and of the process handling under Linux and in the container image.
- Disk space and speed of the hosted temporary folder, and whether it is shared with other data.
- Behaviour with many simultaneous callers beyond the tests (health stayed within a few milliseconds during a
  real fit on a developer machine; that says nothing about a small hosted machine).
- The cost of Python and package start-up per worker (a few seconds on the development machine) on the hosted
  machine, which sets the floor for every job.
