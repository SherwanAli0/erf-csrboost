# First-party usage statistics

Author: Sherwan Ali.

The private owner dashboard keeps application usage separate from research evidence.
It uses the existing SQLite database and owner key. No external analytics service,
new paid resource, visitor cookie or cross-day identifier is introduced.

## What the numbers mean

- Page requests count GET requests to `/`, `/index.html` and `/classic`. Reloads
  count again; HEAD probes, health checks and static assets do not count.
- Visitor-days count distinct daily salted visitor tags among those requests.
  The same person can receive another tag on another day, device or network.
  These are estimates, not distinct people or verified recruiters.
- Jobs accepted count successful HTTP 202 submissions, not completed training.
- Report retrievals count successful result requests, not unique completed jobs.
- Report downloads count successful aggregate export requests, including repeats.
- Legacy analysis attempts retain their original meaning: logged before fitting,
  including attempts which fail later. New data-tool response counts are separate.
- Main counts exclude explicitly marked owner, authenticated test and likely-bot
  traffic. The remainder is unclassified, not confirmed human traffic.
- Date filters use UTC calendar days: today plus the preceding 6, 29 or 89 days,
  or all time. Old code showed the last 30 *active dates* rather than 30 days.

The server records workbench uploads, accepted jobs, result retrievals, exports and
the three integrated data-tool responses. It does not track opening example
reports, tab reading, time on page or browser-only prediction evaluation. The
browser-only file and predictions remain entirely in the browser.

## Link attribution

Use different public destinations for different placements:

```text
LinkedIn Featured:
https://fixyourdataset.fly.dev/?utm_source=linkedin&utm_medium=social&utm_campaign=portfolio&utm_content=featured

ML engineer CV:
https://fixyourdataset.fly.dev/?utm_source=cv&utm_medium=resume&utm_campaign=portfolio&utm_content=ml-engineer

Python backend CV:
https://fixyourdataset.fly.dev/?utm_source=cv&utm_medium=resume&utm_campaign=portfolio&utm_content=python-backend

Data science CV:
https://fixyourdataset.fly.dev/?utm_source=cv&utm_medium=resume&utm_campaign=portfolio&utm_content=data-science

AI engineer CV:
https://fixyourdataset.fly.dev/?utm_source=cv&utm_medium=resume&utm_campaign=portfolio&utm_content=ai-engineer
```

Only the fixed vocabulary in `backend/usage.py` is accepted. No person's name,
email, company, job application identifier, arbitrary query parameter or URL
fragment is retained. The browser carries recognised labels in memory to
same-origin API calls through `X-FYD-Attribution`, without persistent storage.
The original document referrer can supply a recognised source when no tag exists.

Source evidence is labelled `tagged-link`, `page-context`, `referrer` or `unknown`.
`page-context` means the browser carried a public source label from its landing
page, which may have come from a tag or recognised referrer. These are untrusted
client signals, not authentication or proof of a person's journey. Forwarded links
retain their tags. Without tags or referrers the source remains direct / unknown;
historical traffic cannot reliably be reconstructed as CV, LinkedIn or tests.

Updating LinkedIn/CV destinations is a separate owner publication step. This code
does not edit those documents or account links automatically.

## Owner and testing traffic

The existing `/whoami/<owner-key>` page marks the owner's browser. It must be used
on each browser/device. The dashboard itself does not count a visit.

Automated acceptance checks send both `X-FYD-Traffic: test` and a valid
`X-Stats-Key` header. A test marker without the owner key has no effect. Never
place that key in a public link, source code, notebook or screenshot. Prefer HEAD
for uptime checks. User agents matching known bot, crawler, headless or common
command-line clients are labelled `likely-bot`; this heuristic misses some
automation and can misclassify traffic. It does not identify every assistant visit.

Existing rows are preserved and labelled historical where metadata was absent.
No historical events are deleted or silently reclassified as owner/test traffic.

## Storage and boundaries

The migration adds columns with defaults and an index, preserving old columns,
rows, owner flags and analytics history. New referrers are reduced to hostnames;
queries, paths, user information and owner-key URLs are omitted. Historical
referrers can still contain the previously stored URL text. A fixed route
template replaces dataset/job IDs. Status codes and fixed operation names are
recorded, never request bodies, column names, row values or capabilities.

Raw IP addresses and user-agent strings are used transiently for the existing
daily salted tag; the new device/browser families are coarse estimates. Brave
usually appears as Chromium. Tags remain pseudonymous, not anonymous. A configured
`STATS_KEY` is required on the real deployment; its existing value is preserved.
Retention is unchanged and there is no automatic purge. Queries aggregate in
SQLite and return bounded tables. Optional event logging fails open, preserving
application responses when analytics storage fails.

The private dashboard supports `?days=7`, `30`, `90` or `all`. Its normal owner
bookmark uses `/stats/<owner-key>` so relative range links retain authentication.
When authenticating with a header or `?key=`, supply that authentication again on
each request. The dashboard never echoes the key into its generated HTML.
