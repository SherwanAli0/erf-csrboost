# Capability matrix

This describes the upgraded source. Deployment and verification status are recorded separately in
[release verification](RELEASE-VERIFICATION.md). The historical research runner and `/classic` recipes
are preserved. New analysis uses the shared package; it does not regenerate research tables.

| Capability | New interface | Compatibility or limitation |
|---|---|---|
| Explore example | Seeded recorded app run, immediate JSON/HTML and model comparison | Distinct from fictional fixture and research results |
| Saved predictions | Browser-only binary metrics, review budgets, threshold and cost exploration | No upload or model fit; exploration is not fresh independent evidence |
| Dataset analysis | Explicit target/labels/exclusions, bounded asynchronous jobs | Legacy inference remains only in compatibility routes |
| Models | Registry recipes with capacities, seeds and strict failures | Legacy-only samplers/classifiers remain at `/classic` |
| Numeric and categorical data | Training-fitted imputation/scaling/vocabulary, SMOTENC when appropriate | Unseen categories and unsupported widths are reported |
| Splits | Approximately 60/20/20 stratified, group, exact-feature-group and time modes | Repeated 3/20/100-fold legacy UI is not offered as the new procedure |
| Thresholds | Validation-selected, with held-out results | Prediction threshold exploration is labelled exploratory |
| Leakage Lab | Shared package, honest/contaminated samplers on identical original test rows | Demonstration only, not a causal isolation or automated leakage diagnosis |
| Resampling exports | No/random over/random under/SMOTE/clustered SMOTE; explicit row opt-in | Other legacy methods remain separate; no silent fallback in new tools |
| New-row scoring | Logistic, raw forest or ERF; all-labelled-row refit | Fixed 0.5 flag, no refit evaluation claims |
| Row provenance | Partition indices, synthetic flags, duplicate candidate mapping | Ambiguous source patterns are not assigned invented unique origins |
| Aggregate exports | JSON, standalone HTML, effective config and replay bundle | Raw rows are omitted by default; user supplies data for replay |
| Input conversion | CSV/text/Excel/JSON/Stata/archive parsing through bounded workers | New imports require explicit choice for multiple sheets/members; legacy endpoints retain their old selection |
| Parsing diagnostics | Warnings, skipped rows and format/width refusals | A successful conversion is not proof of complete/valid research data |
| Privacy | Capability-owned dataset/jobs, memory-only browser tokens, expiry/cancel | No accounts or per-visitor fairness; restart loses active work |
| Resource controls | One heavy worker plus one waiting slot, light lane, deadlines/memory/storage caps | POSIX memory sampling is not a hard per-worker OS quota |
| Compatibility | `/classic`, original legacy endpoints, usage statistics | Deliberately different evaluation and fallback semantics |

API contracts: [API](API.md). Model differences: [recipes](RECIPE-DIFFERENCES.md).
Resource bounds: [limits](RESOURCE_LIMITS.md). Process model: [architecture](ARCHITECTURE.md).
