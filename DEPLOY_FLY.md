# Deploying to Fly.io

The app runs on Fly.io as `fixyourdataset` (https://fixyourdataset.fly.dev): one always-on machine in
Frankfurt (`fra`, shared CPU, 1 GB RAM) with an encrypted 1 GB volume mounted at `/data`. It listens on
`0.0.0.0:$PORT` (7860) and exposes `/healthz` for the platform check.

## Requirements
- The Fly CLI, logged in (`fly auth login`).
- Do not create a new app or volume. All commands below use `-a fixyourdataset`.

## Before any update
1. Run the tests from the repository root: `python backend/test_head.py` (and `python backend/test_torture.py`).
2. Save the current state so you can roll back:
   - `fly status -a fixyourdataset`
   - `fly image show -a fixyourdataset` (note the image tag and digest)
   - `fly machine list -a fixyourdataset --json` (keep the machine config)

## Update only the code (recommended)
The live HTTP service has `idle_timeout = 900` on its ports. `fly.toml` now states the same value, but
the safest update keeps every other machine setting untouched by changing only the image:

1. Build and push an image without deploying: `fly deploy -a fixyourdataset --build-only --push --image-label <label>`
2. Update the machine to that image: `fly machine update <machine-id> -a fixyourdataset --image registry.fly.io/fixyourdataset:<label>`
3. Verify: `fly status -a fixyourdataset` shows the machine started with a passing check, then request
   `https://fixyourdataset.fly.dev/healthz` and `https://fixyourdataset.fly.dev/` with GET and HEAD.

## Roll back
`fly machine update <machine-id> -a fixyourdataset --image registry.fly.io/fixyourdataset:<previous-tag>`
using the tag and digest saved before the update.

## Notes
- The build context excludes `data_local/` (local analytics) and Markdown files through `.dockerignore`.
- Usage data lives on the `/data` volume and survives updates. The app keeps no automatic retention
  limit and has no purge job: usage events stay until the volume data is deleted. The README section
  "What happens to uploaded files and server state" says exactly what an event contains.
- `STATS_KEY` must be set as a Fly secret on the app. It protects the owner statistics page (the page
  returns 404 without it) and it is the secret in the visitor hash. If it is not set, the statistics
  page is disabled and the visitor tag is computed with a constant key, which is weaker. Never write
  its value in this repository, in a commit message or in documentation. Changing a Fly secret can
  restart the machine, so treat it like an update (see the Fly documentation for `fly secrets`).
- Fly builds the image from the repository files and serves only the running app. Do not commit
  secrets, volume data or `data_local/`.
