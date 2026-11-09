# Architecture

This file describes how the parts of the web application fit together. The relation between the research runner
and the application's models is in `IMPLEMENTATION_MAP.md`. The evaluation package (`evaluation/`) does the
splits, models, metrics and reports; the application only calls it.

## Service

`backend/api.py` is the production entry point: FastAPI under one Uvicorn worker. The Docker image
installs the shared package wheel and serves its workbench assets. `backend/server.py` remains a preserved
legacy entry point; `/classic` provides its old interface through bounded compatibility routes.
Versioned jobs, integrated tools, routes and errors are in `API.md`; capacity controls and measured scope
are in `RESOURCE_LIMITS.md`. Deployment status and receipts belong in `RELEASE-VERIFICATION.md`.

### Parts

```
backend/
  api.py        app factory create_app(config, coordinator), routes, middleware, main()
  schemas.py    request models (unknown fields refused) and the typed error model
  jobs/         the job coordinator (no web code)
    coordinator.py  datasets, jobs, queue, lanes, ownership, lifetimes, the supervisor thread
    runner.py       child process entry point: python backend/jobs/runner.py <command> <job_dir>
    proc.py         start a child in its own process group or job object, kill and reap the whole tree
    store.py        private storage root, byte accounting, atomic JSON files
    tokens.py       capabilities (random, digest only, constant-time check)
    states.py       job states and the legal transitions
    config.py       JobConfig with provisional defaults and FYD_* overrides
  services/     adapter around the evaluation package (profile, train, report HTML) and the exports
    toolbox.py      explicit-label resampling, Leakage Lab and scoring; optional private row exports
    exports.py      provenance label, effective settings, the reproducibility bundle and its replay command
  engine.py     the legacy analysis engine, used only inside worker processes by the service
```

### Process model

- The web process never fits a model and never parses an upload. Profiling, training and the legacy analyze,
  score, convert and profile calls each run in a child process that the coordinator starts. A fit therefore
  cannot block the event loop, the health routes answer in constant time, and a cancel or a timeout is a real
  kill of the child's whole process tree followed by a reap.
- The coordinator is a plain object with a lock and one supervisor thread, created by `create_app` and started
  and stopped by the application's lifespan. Its methods are synchronous. The routes call the blocking ones from
  a worker thread (`def` endpoints or `run_in_threadpool`), never directly on the event loop. The two health
  handlers are `async def` and touch neither the coordinator nor the disk.
- Two lanes. The heavy lane has one running slot and one waiting slot and carries training jobs and the legacy
  analyze and score calls. The light lane runs one short parsing task at a time (upload profiling, legacy
  profile and convert) with a small waiting line. A full lane answers 429 at once; nothing queues without
  limit.
- A worker watches its parent process and kills its own tree if the parent disappears. On Windows each worker
  also runs inside a job object that ends with the service, so a killed service leaves no orphan.
- The legacy heavy routes are `async` endpoints. They hand the work to the coordinator, then poll for the end
  in short sleeps, so the event loop stays free. If the client disconnects, or the request task is cancelled,
  the route cancels the job and waits for the kill to finish.

### Request flow for a training job

1. `POST /api/v1/datasets`: the body is counted while it streams, stored under the private storage root and
   profiled in a light-lane worker. The response holds the profile and a one-time dataset capability.
2. `POST /api/v1/jobs`: the body is validated by `schemas.JobRequest`, then the coordinator checks ownership,
   the per-dataset and global capacity and the storage cap, writes `request.json` into a new job directory and
   queues the job.
3. The supervisor starts the worker. The worker reads the upload, calls `services.train_for_app` with a progress
   callback, writes `progress.json` after every event and finally `result.json` (or `error.json`), each
   atomically.
4. `GET /api/v1/jobs/{id}` reads the (bounded, sanitised) progress file. `/result` and `/exports/{kind}` read
   the result file. The HTML export is rendered by the package in the web process.
5. The supervisor maps the package's per-model statuses to the job state (`succeeded`, `partial`, `failed`,
   `timed_out`), deletes the job directory at once unless there is a report to keep, and the sweep deletes
   everything after its lifetime.

### Ownership and secrets

A capability is returned once, stored only as a SHA-256 digest and sent only in the `Authorization` header. The
coordinator answers a wrong, missing or foreign capability on an existing id exactly like an unknown id. Error
bodies and log lines carry no capability, path, traceback or table row; the middleware logs only the class name of
an unexpected exception. The owner statistics key of the legacy pages is never logged and is compared in constant
time. Uvicorn's access log is off because it would record URLs.

### Pages and exports

- The workbench is the maintained interface. Its files live in `evaluation/src/rare_event_eval/web` and are the same
  files the standalone local server (`python -m rare_event_eval serve`) serves. `backend/api.py` serves `index.html`
  at `/` and `/index.html` and every other plain file of that folder at its own root path, from an allowlist built
  once at startup (no request text is joined into a path). The page looks for the hosted API with
  `GET /api/v1/version` and shows the "Analyze a dataset" flow when it answers. The old single page
  (`frontend/index.html`) is served unchanged at `/classic` for legacy recipes and receipts; the new interface
  integrates its own explicit-label data tools. The folder can be changed with
  `FYD_WEB_DIR`, which the tests use. A deployment image has to contain that folder: without an `index.html` the service
  answers 503 at `/` instead of falling back silently.
- Exports are built by the coordinator from the stored report and the validated request, in the web process, by
  `services/exports.py` and the package's renderer. The report is annotated with a provenance label (`fresh app run`)
  and an app block when it is read, so `/result` and every export agree. The reproducibility bundle is a pure function of
  the report, the request, the limits and the app version: fixed member order, timestamps and compression level, bounded
  in size, no row of the upload. Its replay command is built from the effective settings and checked by a test that runs
  it with the package command line (see `API.md`).

### Lifetimes and restarts

Datasets and jobs live in a private temporary folder, and their registry is in memory. At start the service
deletes the whole folder (only if it carries the marker file the service writes, or is empty), so after a restart
every earlier id is unknown. Stopping the service cancels every job, kills every worker tree and deletes the folder.

### Tests

`tests/test_jobs.py` covers the coordinator with real child processes. `tests/test_api_v1.py` and
`tests/test_api_compat.py` cover the HTTP layer, including one real `python backend/api.py` process for health
during a fit, a chunked oversize upload, a client disconnect and a clean shutdown. Run all of them from the
repository root with `python -B -m unittest discover -s tests`.
