"""Standalone HTML rendering with no scripts, remote assets or input rows."""
from __future__ import annotations

from html import escape
import json
from copy import deepcopy


SUPPORTED_TRAINING_SCHEMAS = ("training-1.0", "training-1.1")
TRAINING_PROVENANCE_KEYS = ("schema_version", "label_mapping", "predictors", "excluded_columns", "input_parsing", "split",
                            "model_records", "preprocessing", "fallback_policy", "failures", "seed")


def training_report(comparison):
    version = comparison.get('schema_version')
    if version is not None and version not in SUPPORTED_TRAINING_SCHEMAS:
        raise ValueError('Only training report schemas ' + ', '.join(SUPPORTED_TRAINING_SCHEMAS) + ' are supported.')
    successful = {name: row for name, row in comparison['outcomes'].items() if row['status']=='success'}
    if not successful:
        raise ValueError('No successful model is available for an HTML report. Inspect the JSON outcomes.')
    report = deepcopy(next(iter(successful.values()))['report'])
    report['models'] = {name: deepcopy(row['report']['models'][name]) for name,row in successful.items()}
    report['execution_mode'] = 'fresh_training'
    report['policies']['threshold'] = None
    report['policies']['selection'] = 'Each model uses a threshold chosen on validation data only. Test labels were excluded from selection.'
    report['execution'] = {name: {'status':row['status'], 'worker_wall_seconds':row.get('worker_wall_seconds'),
                              **{key:row.get('execution',{}).get(key) for key in ('fit_seconds','peak_process_rss_bytes','split','calibration')}}
                           for name,row in comparison['outcomes'].items()}
    if version == 'training-1.1':
        report['training_provenance'] = {key: deepcopy(comparison[key]) for key in TRAINING_PROVENANCE_KEYS if key in comparison}
    failed = [name for name,row in comparison['outcomes'].items() if row['status']!='success']
    if failed:
        report['limitations'].append('Models without successful results: '+', '.join(failed)+'. Inspect JSON outcomes; these were not silently removed.')
    return report


def _text(value: object) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value), quote=True)


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th scope='col'>{escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def render_html(report: dict) -> str:
    """Render a report returned by evaluate_predictions as an offline HTML file."""
    if report.get("schema_version") != "1.0":
        raise ValueError("Only report schema 1.0 is supported.")
    task, data, policies = report["task"], report["data"], report["policies"]
    metrics_rows = []
    decisions = []
    notes = []
    calibration_sections = []
    for model, result in report["models"].items():
        metric = result["metrics"]
        metrics_rows.append([model, metric["average_precision"], metric["roc_auc"],
                             metric["brier_score"], metric["log_loss"]])
        if result["threshold_result"] is not None:
            row = result["threshold_result"]
            decisions.append([model, f"Threshold {row['threshold']:.6g}", row["flagged"],
                              row["true_positives"], row["false_positives"], row["false_negatives"],
                              row["precision"], row["recall"], row["assumed_total_cost"]])
        for row in result["review_budget_results"]:
            decisions.append([model, f"Top {row['budget']}", row["flagged"], row["true_positives"],
                              row["false_positives"], row["false_negatives"], row["precision"],
                              row["recall"], row["assumed_total_cost"]])
            tie = row["boundary_tie"]
            if tie is not None:
                notes.append(f"{model}, top {row['budget']}: score tie at the cutoff. "
                             f"Possible positives found range from {tie['true_positives_min']} "
                             f"to {tie['true_positives_max']}; these are tie-allocation bounds, "
                             "not confidence intervals.")
        notes.extend(f"{model}: {warning}" for warning in result["warnings"])
        if result["calibration"] is not None:
            rows = [[f"{row['lower']:.2f} to {row['upper']:.2f}", row["count"], row["mean_score"],
                     row["observed_positive_fraction"]] for row in result["calibration"]["bins"]]
            calibration_sections.append(f"<h3>{escape(model)}</h3>" + _table(
                ["Score interval", "Rows", "Mean score", "Observed positive fraction"], rows))
    metric_table = _table(["Model", "Average precision", "ROC AUC", "Brier score", "Log loss"], metrics_rows)
    decision_table = (_table(["Model", "Fixed policy", "Flagged", "Positives found", "False alarms",
                              "Positives missed", "Precision", "Recall", "Assumed cost"], decisions)
                      if decisions else "<p>No decision policies were supplied.</p>")
    warning_html = ("<ul>" + "".join(f"<li>{escape(note)}</li>" for note in notes) + "</ul>"
                    if notes else "<p>No additional descriptive warnings.</p>")
    limitations = "".join(f"<li>{escape(note)}</li>" for note in report["limitations"])
    checks = _table(["Check", "Status"], [[key, value] for key, value in report["checks"].items()])
    assumptions = policies["cost_assumptions"]
    costs_html = (f"<pre>{escape(json.dumps(assumptions, indent=2, allow_nan=False))}</pre>"
                  if assumptions is not None else "<p>No cost assumptions supplied. Cost results are unavailable.</p>")
    calibration_html = "".join(calibration_sections) or "<p>Unavailable for ranking scores.</p>"
    provenance = {key: report[key] for key in ("schema_version", "tool", "created_utc", "task", "data", "policies")}
    if 'execution' in report:
        provenance['execution'] = report['execution']
    if 'training_provenance' in report:
        provenance['training_provenance'] = report['training_provenance']
    provenance_html = escape(json.dumps(provenance, indent=2, allow_nan=False))
    execution_note = ("Fresh models were trained using separate training, validation and test partitions."
                      if report.get("execution_mode") == "fresh_training" else
                      "Saved predictions evaluated locally. No model was trained.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Rare-event evaluation report</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#192738;background:#f3f6fa;line-height:1.55}}
body{{max-width:1120px;margin:32px auto;padding:0 20px}}header,section{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:24px;margin-bottom:18px}}
h1,h2,h3{{line-height:1.2}}h1{{margin-top:8px}}h2{{font-size:1.25rem}}.eyebrow{{color:#315f8b;font-weight:700;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase}}
.summary{{display:flex;gap:24px;flex-wrap:wrap}}.summary strong{{font-size:1.6rem;display:block}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.92rem}}
th,td{{border-bottom:1px solid #dce4ed;text-align:left;padding:10px;vertical-align:top}}th{{background:#eef4fa}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:16px;font-size:.85rem}}
li{{margin:8px 0}}.muted{{color:#506176}}@media print{{body{{background:white;margin:0;max-width:none}}header,section{{break-inside:avoid}}}}
</style></head><body>
<header><div class="eyebrow">Sherwan Ali | rare-event-eval {_text(report['tool']['version'])}</div>
<h1>Rare-event evaluation</h1><p>{execution_note}</p>
<p>{escape(data.get('description') or 'No dataset description supplied.')} <span class="muted">(User-declared description.)</span></p>
<div class="summary"><div><strong>{data['rows']}</strong>evaluated rows</div><div><strong>{data['positive_rows']}</strong>positive rows</div>
<div><strong>{data['positive_prevalence']:.2%}</strong>positive prevalence</div></div>
<p>Positive label: <strong>{escape(task['positive_label'])}</strong>. Negative label: <strong>{escape(task['negative_label'])}</strong>.</p>
<p class="muted">Score kind: {escape(task['score_kind'])}. Evaluation role: {escape(task['evaluation_role'])} (user-declared, unverified).</p></header>
<section><h2>Model scores</h2>{metric_table}<p class="muted">Average precision is not trapezoidal PR area. Compare models on the same population; prevalence affects interpretation. Lower Brier score and log loss are better.</p></section>
<section><h2>Decision policies</h2>{decision_table}<p class="muted">Thresholds use score &gt;= threshold. Top-k policies break ties by input row order. {escape(policies['selection'])} No automatic winning model is declared.</p></section>
<section><h2>Warnings and score ties</h2>{warning_html}</section>
<section><h2>Descriptive reliability bins</h2><p>These bins do not certify calibration. Intervals include their lower endpoint; only the last includes its upper endpoint. Small bins can be noisy.</p>{calibration_html}</section>
<section><h2>Cost assumptions</h2>{costs_html}</section>
<section><h2>Checks and limitations</h2>{checks}<ul>{limitations}</ul></section>
<section><h2>Configuration and provenance</h2><pre>{provenance_html}</pre><p class="muted">This report contains aggregates, not raw prediction rows. Share it deliberately; aggregates can still reveal sensitive information.</p></section>
</body></html>"""
