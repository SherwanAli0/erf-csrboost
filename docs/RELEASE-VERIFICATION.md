# Release verification

Author: Sherwan Ali.

This document records the scope of checks for the FixYourDataset upgrade. Source review, local
verification, public CI and deployment are separate gates. The working branch is
`feat/fyd-flagship-upgrade`; an unmerged branch is not a live release.

## Locally observed

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

Full suite counts, clean-package results and CI/deployment references are added after their actual
completion. A blocked test or skipped check must not be reported as passing.

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

Package distributions are intended for a tagged GitHub release, not a claimed PyPI publication.
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
