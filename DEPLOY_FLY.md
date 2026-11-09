# Deployment and rollback

FixYourDataset runs at https://fixyourdataset.fly.dev on the existing Frankfurt Fly application,
one shared CPU, 1 GB memory and its encrypted 1 GB `/data` volume. Deployment is manual and changes
the image only. Do not create an app, machine, volume or paid builder, or change machine size,
region, secrets, always-on behavior, concurrency or timeout settings.

## Prepare

1. Complete source review, tests, clean package installation, public CI and the constrained image check.
2. Merge the reviewed public PR normally. Synchronize only reviewed release files into the private
   deployment checkout; preserve private-only settings and compare shared file hashes.
3. Read `fly machine list -a fixyourdataset --json`, `fly image show -a fixyourdataset` and
   `fly volumes list -a fixyourdataset --json`. Save a private redacted configuration receipt and the
   previous immutable image for rollback. Never publish the complete environment or credentials.
4. Confirm no machine command overrides the image CMD. Do not delete data or older images.

## Local build and image-only update

The image installs the locked runtime and package wheel and starts `python backend/api.py` on 7860.
Use the exact reviewed public source commit as the public release id:

```bash
docker build --build-arg FYD_RELEASE_ID=<reviewed-public-commit> -t registry.fly.io/fixyourdataset:<release-label> .
python -B tools/container_smoke.py --image registry.fly.io/fixyourdataset:<release-label>
fly auth docker
docker push registry.fly.io/fixyourdataset:<release-label>
fly machine update <existing-machine-id> -a fixyourdataset --image registry.fly.io/fixyourdataset@sha256:<pushed-digest> --yes
```

The smoke overrides the source id with `container-check` only in its disposable test container.
Build locally to avoid provisioning a paid remote builder. Do not run an unrestricted `fly deploy`
that could reconcile unrelated settings. Do not change existing Fly secrets for this release.

## Acceptance

Confirm Fly checks, GET/HEAD on `/`, `/healthz`, assets and `/classic`, and GET `/api/v1/version`.
The latter must identify the reviewed source and package/app versions. Explore the example without
training; check small fictional numeric and mixed jobs, exports, foreign-capability denial,
cancellation and integrated tools. Compare resource/service/mount/environment configuration before
and after without printing secret values. Preserve `/data`, its analytics and existing secret setup.

## Rollback

If readiness, core journeys, privacy or numerical consistency fail, or the service cannot remain
healthy within its allocation, update the same machine to the saved immutable image:

```bash
fly machine update <existing-machine-id> -a fixyourdataset --image <saved-previous-image> --yes
```

Verify health and the prior basic journey. Do not delete `/data` or create a replacement volume.
There is no destructive analytics migration. Record failure evidence and continue the fix locally;
rolling back does not make a failed release complete.

## Storage

The narrow build context excludes research data, Git history, local environments, analytics and
private evidence. Only explicit runtime source/assets, samples and the package wheel enter the image.
The existing analytics remains at `/data/usage.db`; job storage uses a separate temporary root.
`STATS_KEY` stays in Fly secrets and must never appear in a repository or command output.
See [resource limits](docs/RESOURCE_LIMITS.md) and [verification](docs/RELEASE-VERIFICATION.md).
