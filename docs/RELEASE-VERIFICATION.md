# Release verification

Author: Sherwan Ali.

This document records actual source, CI, image and live checks. Historical preparation notes
below are retained with their original scope; the shipped-release record takes precedence.

## Shipped release, 10 November 2026

- App 2.0.1 and package 0.3.1 are deployed. Runtime source is `2d26897`; `c118045`
  changes only a cancellation regression test. [PR #2](https://github.com/SherwanAli0/erf-csrboost/pull/2)
  merged as `97c4c8e`. [Release and distributions](https://github.com/SherwanAli0/erf-csrboost/releases/tag/fyd-v2.0.1).
- [All five CI jobs passed](https://github.com/SherwanAli0/erf-csrboost/actions/runs/42793611554):
  Linux/Windows and Python 3.11/3.12 package/application suites, browser parity, legacy
  regressions, distribution builds, clean wheel installation and the constrained Linux image.
- Local application suite: 260 checks, two skips. Package suite: 271 checks, two skips.
  Nine focused usage checks, nine browser-client checks and 18 browser static checks passed.
  The later cancellation-test synchronization change passed its targeted check and final CI.
- The exact deployed image passed 49 local checks at one CPU and 1,024 MiB, including
  the seeded 10,000-row case. This is bounded fixture evidence, not a guarantee for every input.
- The earlier 2.0 release passed 35 small live acceptance checks. The 2.0.1 follow-up
  verified all four private dashboard date ranges and 12 real action-accounting checks,
  including one owned 160-row fictional logistic comparison, report/export and cleanup.
  These requests were authenticated as test traffic and excluded from the main counts.
- A backup and checksum comparison preserved all original columns of 189 historical usage
  rows. Existing machine allocation, volume, environment and service settings were unchanged.
  No new resources, external analytics service or visitor cookie were introduced.

The 2.0.1 image platform digest is
`sha256:4736357adebd3eff4f71cc183074d6e9b71c7bec7fb2caa9ca3ec5374e1ec84f`.
Detailed owner receipts remain private because they include operational/account context.
[Usage counting and attribution limits](USAGE-STATS.md) describe what the dashboard can establish.

## Earlier local preparation receipts

- Python 3.11.9 application environment: `pip check` passed with the committed dependency lock.
- Integrated-tool contract and real-worker API tests: 10 passed, covering opt-in row exports,
  original holdout cells, disjoint partitions, synthetic mapping, majority positive labels,
  scoring-column collisions, common Leakage Lab test rows, spreadsheet safety and input refusals.
- Isolated Chrome run: seven routes at 375 px without page overflow; unique document ids;
  independent prediction/hosted cost fields; prediction evaluation without a data request;
  resampling, five requested CSV downloads, Leakage Lab, scoring and cancellation. No page errors.
- Production-image smoke: 44 checks passed at exactly one CPU and 1,024 MiB, including three
  repeated comparisons, four aggregate exports, foreign-capability refusal, full-queue 429,
  health during fitting, cancellation, integrated resampling and oversized GZIP refusal.
  Thirteen `docker stats` samples peaked at 335.6 MiB. This sampled peak is not a hard maximum,
  a maximum-size-input benchmark or a measurement on the Fly host.
- Historical research training was not rerun. Result files and submitted research remain unchanged.

The following paragraphs preserve the earlier incremental checks. The final release gates above
supersede their provisional status. A blocked test or skipped check is not a passing check.

The full local application suite passed 244 tests (two platform-specific skips); the package suite
passed 271 (two optional skips). Later explicit-import coverage added six passing checks and the
48 service tests passed again. Final public CI checks the final combined source.

The larger local image run passed 49 checks, including 10,000 rows and 100 expanded features with
logistic regression and ERF both successful, plus concurrent light conversion. Thirty memory samples
peaked at 431.5 MiB. This covers that particular seeded fixture, not every table within the limits.
The built wheel installed in a fresh Python 3.11 environment outside the source tree, passed
`pip check` and generated JSON/HTML reports from the fictional prediction CSV.

The first public matrix exposed a Windows-only newline assertion and cross-platform regeneration
checks that treated the Python-version field and last-bit floating-point differences as changes in
results. The test now accepts either newline style, checks environment metadata separately and allows
at most 1e-12 absolute numerical drift; labels, partitions, counts, keys and policies remain exact.
This is a verification correction, not a change to model results. Actual CI status remains separate.

## Repeat the checks

Install the application and build locks, then the package as described in the root README.

```bash
python -B -m unittest discover -s tests
cd evaluation
python -B -m unittest discover -s tests
cd ..
node evaluation/tests/test_browser_parity.js
python -B backend/test_head.py
python -B backend/test_rare_split_guard.py
python -B backend/test_sample_cards.py
python -B backend/test_small_fixes.py
python -B backend/test_torture.py
node frontend/test_receipt.js
node frontend/test_verdict.js
python -m build --no-isolation --outdir dist ./evaluation
docker build -t fyd-check .
python -B tools/container_smoke.py --image fyd-check
```

Package distributions are attached to the tagged GitHub release linked above. They are not published on PyPI.
The CI workflow checks Linux and Windows with Python 3.11/3.12, a fresh wheel install outside the
source tree and the constrained Linux image. A defined matrix alone is not proof that it passed.

## Remaining operational limits

One shared heavy slot and one waiting slot provide bounded work, not per-visitor fairness. The
POSIX worker limit samples RSS; the container memory limit is the hard outer boundary. The
light lane can run beside a heavy worker. Large, wide and hostile inputs can be refused or hit
deadlines. Startup imports, conversion and exports consume capacity too.

New tools accept bounded binary CSVs and stratified partitions. They do not expose every format,
resampler or repeated-fold option of the previous interface. Scoring outputs are not calibrated
probabilities by assertion; Leakage Lab results are not evidence about a user's real pipeline.

Dataset/job capabilities are bearer secrets held in memory; no account recovery is offered.
Restarts interrupt jobs and lose access. Usage statistics remain pseudonymous and persistent on
the existing volume. The app is a demonstration with explicit limits, not a regulated-data platform.
