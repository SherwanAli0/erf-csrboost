/* Sherwan Ali. Pure rendering and formatting: report sections, provenance labels, comparison wording, partial and failed
   models, and escaping of hostile text in every place a label, column name or message is printed. */
'use strict';
const assert = require('node:assert/strict');
const S = require('./web_support.js');
const { U, C, RV, V, FLOW, M, P } = S.load();
const { test } = S;

const ch = n => String.fromCharCode(n);
const clone = x => JSON.parse(JSON.stringify(x));
const result = () => clone(S.fixture('web-hosted-result.json'));
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&#96;/g, '`').replace(/&amp;/g, '&').replace(/\s+/g, ' ');

test('number formatting is explicit about unavailable values and signs', () => {
  assert.equal(U.fixed(null), 'Unavailable');
  assert.equal(U.fixed(NaN), 'Unavailable');
  assert.equal(U.fixed(0.5234, 3), '0.523');
  assert.equal(U.signed(-0.0089, 3), '-0.009');
  assert.equal(U.signed(0.0004, 3), '0.000', 'a value that rounds to zero has no sign');
  assert.equal(U.signed(0.0123, 3), '+0.012');
  assert.equal(U.pct(0.2906), '29.1%');
  assert.equal(U.pct(undefined), 'Unavailable');
  assert.equal(U.int(1234567), '1,234,567');
  assert.equal(U.duration(30), '30 seconds');
  assert.equal(U.duration(1800), '30 minutes');
  assert.equal(U.megabytes(8388608), '8.0 MB');
  assert.ok(!/[^\x00-\x7F]/.test(U.signed(-1) + U.pct(0.1) + U.int(5)), 'formatted numbers are ASCII');
});

test('file sizes use bytes, KB or MB, never 0.0 MB for a small file', () => {
  assert.equal(U.size(45678), '45 KB');
  assert.equal(U.size(5000), '4.9 KB');
  assert.equal(U.size(0), '0 bytes');
  assert.equal(U.size(1), '1 byte');
  assert.equal(U.size(1023), '1023 bytes');
  assert.equal(U.size(2621440), '2.5 MB');
  assert.equal(U.size(8388608), '8.0 MB');
  assert.equal(U.size(null), 'Unavailable');
  assert.ok(!U.size(45678).startsWith('0.0'));
});

test('download names are meaningful, ASCII and never empty', () => {
  assert.equal(U.downloadName('report_html', 'liver_disease.csv', '2026-11-09T19:37:36Z'), 'fyd-liver-disease-2026-11-09-report.html');
  assert.equal(U.downloadName('repro_bundle', 'Loans 2026 (final).xlsx', '2026-11-09'), 'fyd-loans-2026-final-2026-11-09-repro-bundle.zip');
  assert.equal(U.downloadName('config_json', '../../etc/passwd', undefined), 'fyd-etc-passwd-undated-config.json');
  assert.equal(U.downloadName('report_json', '' + ch(0x00E9) + '' + ch(0x00E8) + '.csv', '2026-01-02'), 'fyd-analysis-2026-01-02-report.json');
  assert.throws(() => U.downloadName('rows_csv', 'x', '2026-01-01'));
});

test('a hosted report shows dataset and split before any metric, then each metric family separately', () => {
  const html = RV.renderTrainingReport(result(), { provenance: 'fresh_app_run', idPrefix: 't' });
  const order = ['Dataset, positive class and split', 'Models in this run', 'Ranking quality', 'Threshold outcomes', 'Probability diagnostics', 'Time and resource cost', 'Limits of this report'].map(h => html.indexOf(h));
  assert.ok(order.every(i => i > 0), 'every section is present');
  assert.deepEqual([...order].sort((a, b) => a - b), order, 'sections are in reading order');
  const t = text(html);
  assert.match(t, /Positive class 1 \(chosen explicitly, never inferred\)/);
  assert.match(t, /train 349 100 249/);
  assert.match(t, /the same partitions were used for every model: yes/);
  assert.match(t, /Threshold \(from validation\)/);
  assert.match(t, /maximizes F1 on the validation partition/);
  assert.match(t, /maximizes F1 on the validation partition 117 no/);
  assert.match(t, /Peak process memory/);
});

test('the provenance label is printed as text, with its explanation, for each of the four kinds', () => {
  for (const key of Object.keys(C.PROVENANCE)) {
    const html = RV.renderTrainingReport(result(), { provenance: key, idPrefix: 'p' });
    assert.ok(html.includes(C.PROVENANCE[key].label), key);
    assert.ok(html.includes(`data-provenance="${key}"`), key);
  }
  assert.deepEqual(Object.values(C.PROVENANCE).map(p => p.label), ['fictional fixture', 'recorded app run', 'fresh app run', 'historical research record']);
  // the report JSON may carry the label text instead of the key
  assert.equal(RV.provenanceKey('recorded app run'), 'recorded_app_run');
  assert.ok(RV.renderTrainingReport(result(), { provenance: 'recorded app run' }).includes('recorded app run'));
});

test('no green winner badge and no winner wording; differences are signed and have context', () => {
  const html = RV.renderTrainingReport(result(), { provenance: 'fresh_app_run' });
  const t = text(html).toLowerCase();
  assert.ok(!/winner|best model|\bwins\b|recommended/.test(t.replace('never named the winner automatically', '').replace('no automatic winner', '')), 'no winner wording beyond the statements that deny one');
  assert.match(t, /no automatic winner/);
  assert.match(t, /never named the winner automatically/);
  assert.match(t, /difference from reference/);
  assert.match(t, /near-tied with the highest|highest in this run/);
  assert.ok(!/class="[^"]*(winner|best|green)[^"]*"/.test(html), 'no winner or green class names');
});

function synthetic(models) {
  // models: [{id, cost, ap, auc, fit}] -> a training-1.1 result built from the real fixture shape
  const base = result(), template = base.outcomes.logistic;
  const outcomes = {};
  models.forEach(m => {
    const o = clone(template);
    o.model_record.display_name = m.name || m.id; o.model_record.cost_class = m.cost; o.model_record.recipe_id = `rare_event_eval/${m.id}/1`;
    o.report.models = { [m.id]: Object.assign(clone(template.report.models.logistic), { metrics: { average_precision: m.ap, roc_auc: m.auc, brier_score: 0.2, log_loss: 0.5 } }) };
    o.execution.fit_seconds = m.fit; o.execution.model = m.id;
    outcomes[m.id] = o;
  });
  base.outcomes = outcomes; base.model_records = Object.fromEntries(Object.entries(outcomes).map(([k, v]) => [k, v.model_record]));
  return base;
}

test('the comparison statement says plainly when the simpler model is competitive', () => {
  const close = RV.comparisonStatement(RV.trainingViewModel(synthetic([{ id: 'logistic', name: 'Logistic regression', cost: 'light', ap: 0.7, auc: 0.8, fit: 0.1 }, { id: 'erf', name: 'ERF', cost: 'heavy', ap: 0.706, auc: 0.81, fit: 9 }])).models);
  assert.equal(close.kind, 'simpler_close');
  assert.match(close.text, /simpler model Logistic regression is competitive/);
  assert.match(close.text, /not a statistical test/);
  const better = RV.comparisonStatement(RV.trainingViewModel(synthetic([{ id: 'logistic', cost: 'light', ap: 0.75, auc: 0.8, fit: 0.1 }, { id: 'erf', cost: 'heavy', ap: 0.7, auc: 0.8, fit: 9 }])).models);
  assert.equal(better.kind, 'simplest_best');
  assert.match(better.text, /no heavier model improved on it/);
  const clear = RV.comparisonStatement(RV.trainingViewModel(synthetic([{ id: 'logistic', cost: 'light', ap: 0.6, auc: 0.7, fit: 0.1 }, { id: 'erf', cost: 'heavy', ap: 0.75, auc: 0.8, fit: 9 }])).models);
  assert.equal(clear.kind, 'heavier_clear');
  assert.match(clear.text, /confirm it on other data/);
  assert.equal(RV.comparisonStatement(RV.trainingViewModel(synthetic([{ id: 'logistic', cost: 'light', ap: 0.6, auc: 0.7, fit: 0.1 }])).models).kind, 'single');
  assert.equal(RV.comparisonStatement([]).kind, 'none');
  // exactly at the display rule: a gap of 0.01 is not "near-tied"
  assert.equal(RV.comparisonStatement(RV.trainingViewModel(synthetic([{ id: 'logistic', cost: 'light', ap: 0.7, auc: 0.8, fit: 0.1 }, { id: 'erf', cost: 'heavy', ap: 0.71, auc: 0.8, fit: 9 }])).models).kind, 'heavier_clear');
});

test('the real fixture run says the simpler model is competitive', () => {
  const st = RV.comparisonStatement(RV.trainingViewModel(result()).models);
  assert.equal(st.kind, 'simpler_close');
  assert.ok(text(RV.renderTrainingReport(result())).includes('is competitive'));
});

test('partial results show failed and timed-out models beside successes with reason codes', () => {
  const r = result();
  r.outcomes.rf_weighted = { status: 'timeout', model: 'rf_weighted', timeout_seconds: 60, worker_wall_seconds: 60.2, reason_code: 'MODEL_TIMEOUT', model_record: r.outcomes.rf_weighted.model_record };
  r.outcomes.erf = { status: 'error', model: 'erf', error: 'SMOTE requires at least two minority rows', reason_code: 'RESAMPLING_INFEASIBLE', model_record: Object.assign(clone(r.outcomes.logistic.model_record), { id: 'erf', display_name: 'ERF (20 bags x 50 trees)', recipe_id: 'rare_event_eval/erf/1' }) };
  const html = RV.renderTrainingReport(r, { provenance: 'fresh_app_run' }), t = text(html);
  assert.match(t, /Partial result: 1 of 3 models produced a result/);
  assert.match(t, /Models without a result/);
  assert.match(t, /MODEL_TIMEOUT/);
  assert.match(t, /RESAMPLING_INFEASIBLE/);
  assert.match(t, /reached its time limit/);
  assert.match(t, /Time limit: 60\.0 s/);
  assert.match(t, /SMOTE requires at least two minority rows/);
  assert.match(t, /were not dropped, and no other recipe was substituted/);
  assert.match(t, /Logistic regression Result produced/, 'the success is listed in the same table');
  assert.match(t, /Timed out/);
  assert.deepEqual(S.tagAudit(html, []), []);
});

test('a report with no successful model does not throw and says nothing was compared', () => {
  const r = result();
  Object.keys(r.outcomes).forEach(k => { r.outcomes[k] = { status: 'error', model: k, error: 'boom', reason_code: 'MODEL_FIT_ERROR', model_record: r.outcomes[k].model_record }; });
  r.split.achieved = null; r.split.note = 'No model completed, so no achieved split is recorded.';
  const t = text(RV.renderTrainingReport(r));
  assert.match(t, /No achieved split is recorded because no model completed/);
  assert.match(t, /MODEL_FIT_ERROR/);
});

test('costs appear only when supplied and always as assumptions', () => {
  const none = text(RV.renderTrainingReport(result()));
  assert.match(none, /No cost assumptions were supplied, so no cost is shown/);
  const r = result();
  Object.values(r.outcomes).forEach(o => { Object.values(o.report.models).forEach(m => { m.threshold_result.assumed_total_cost = 123.5; }); });
  const t = text(RV.renderTrainingReport(r));
  assert.match(t, /Assumed cost/);
  assert.match(t, /123\.5/);
  assert.match(t, /These are assumptions, not measured savings/);
});

test('ties at a review budget are reported as allocation bounds, not intervals', () => {
  const r = result();
  const m = r.outcomes.logistic.report.models.logistic;
  m.review_budget_results[0].boundary_tie = { score: 0.3, tied_rows: 4, available_slots: 2, true_positives_min: 1, true_positives_max: 3 };
  assert.match(text(RV.renderTrainingReport(r)), /tied scores can yield 1 to 3 positive cases found.*not confidence intervals/);
});

test('few positives in the test partition raise a caution', () => {
  const r = result();
  Object.values(r.outcomes).forEach(o => { o.report.data.positive_rows = 7; });
  assert.match(text(RV.renderTrainingReport(r)), /only 7 positive cases/);
});

test('probability diagnostics are shown with their limits, and absent when not valid', () => {
  assert.match(text(RV.renderTrainingReport(result())), /do not certify calibration/);
  const r = result();
  Object.values(r.outcomes).forEach(o => Object.values(o.report.models).forEach(m => { m.metrics.brier_score = null; m.metrics.log_loss = null; }));
  assert.match(text(RV.renderTrainingReport(r)), /Probability diagnostics are not available/);
});

test('tables sit in labelled, focusable scroll regions with captions; ids are unique', () => {
  const html = RV.renderTrainingReport(result(), { idPrefix: 'z' });
  const regions = [...html.matchAll(/<div class="table-scroll" role="region" aria-labelledby="([^"]+)" tabindex="0"><table><caption id="([^"]+)">/g)];
  assert.ok(regions.length >= 8);
  regions.forEach(m => assert.equal(m[1], m[2]));
  assert.equal((html.match(/<table>/g) || []).length, regions.length, 'every table is inside a region');
  assert.deepEqual(S.duplicateIds(html), []);
  assert.ok((html.match(/<th scope="col"/g) || []).length > 20);
  assert.ok(html.includes('<th scope="row">'));
});

test('charts carry a text summary, printed values and a baseline key', () => {
  const html = RV.renderTrainingReport(result());
  assert.match(html, /<figure class="chart"[^>]*><figcaption[^>]*>Average precision on the test partition<\/figcaption>/);
  assert.match(html, /no-skill baseline \(positive share 29\.1%\)/);
  assert.match(html, /<svg class="bar" viewBox="0 0 100 14"[^>]*role="img" aria-label="Logistic regression: 0\.524"/);
  assert.match(html, /<p class="chart-summary">Average precision ranges from 0\.524 to 0\.534/);
  assert.match(html, /<span class="bar-value">0\.524<\/span>/);
});

/* ---- escaping ---- */
const PAYLOADS = [
  '<script>alert(1)</script>', '"><img src=x onerror=alert(1)>', "' onfocus='alert(1)' autofocus='", '</td></tr></table><iframe src=//evil></iframe>',
  '<svg onload=alert(1)>', '&lt;b&gt; & &amp;', '=cmd|\' /C calc\'!A0', '`${alert(1)}`', 'javascript:alert(1)'
];
function hostile(r, payload) {
  const m = r.label_mapping;
  m.positive_label = payload; m.negative_label = payload + 'N'; m.negative_labels = [payload + 'N', payload + 'M'];
  m.original_labels = [{ label: payload, count: 167, internal: 1 }, { label: payload + 'N', count: 416, internal: 0 }];
  r.predictors = [payload + 'p1', payload + 'p2']; r.predictor_kinds = { [payload + 'p1']: 'numeric', [payload + 'p2']: payload };
  r.excluded_columns = [{ column: payload, role: payload, reason: payload }];
  r.input_parsing.details.format_notes = [payload]; r.input_parsing.details.read_warnings = [payload]; r.input_parsing.details.note = payload;
  r.input_parsing.parser = payload; r.input_sha256 = payload;
  Object.values(r.outcomes).forEach(o => {
    o.model_record.display_name = payload; o.model_record.recipe_id = payload; o.model_record.cost_class = 'moderate';
    o.report.tool = { name: payload, version: payload }; o.report.created_utc = payload;
    Object.values(o.report.models).forEach(mm => { mm.warnings = [payload]; mm.calibration.interpretation = payload; });
    o.execution.warnings = [payload];
    o.execution.split.interpretation = payload;
  });
  r.outcomes.erf = { status: 'error', model: 'erf', error: payload, reason_code: payload, timeout_seconds: 5, model_record: { display_name: payload, recipe_id: payload, cost_class: 'heavy' } };
  r.split.requested.mode = payload;
  return r;
}
test('hostile text is escaped in every part of a training report', () => {
  for (const payload of PAYLOADS) {
    const html = RV.renderTrainingReport(hostile(result(), payload), { provenance: 'fresh_app_run', idPrefix: 'h' });
    assert.deepEqual(S.tagAudit(html), [], `payload ${payload}`);
    assert.ok(!html.includes(payload) || !/[<>"'`]/.test(payload.replace(/&[a-z]+;/g, '')) , 'raw payload does not appear');
    assert.ok(!html.includes('<script') && !html.includes('<img') && !html.includes('<iframe'), payload);
    assert.ok(text(html).includes(payload), 'the text is still readable after escaping: ' + payload);
  }
});
test('hostile text is escaped in a saved-prediction report and its standalone document', () => {
  const parsed = M.fromCSV('y,s\nA,.9\nB,.1\nA,.8\nB,.2\n', 'y', ['s']);
  for (const payload of PAYLOADS) {
    const labels = parsed.labels.map(l => l === 'A' ? payload : payload + 'N');
    const scores = { [payload + 'col']: parsed.scores.s };
    const report = M.evaluate(labels, scores, { positiveLabel: payload, negativeLabel: payload + 'N', scoreKind: 'probability', reviewBudgets: [1], threshold: 0.5, description: payload });
    report.data.description = payload; report.policies.selection = payload;
    const html = RV.renderPredictionReport(report, { provenance: 'fresh_app_run', provenanceNote: payload, cannotCertify: payload, idPrefix: 'q' });
    assert.deepEqual(S.tagAudit(html), [], payload);
    assert.ok(text(html).includes(payload));
    const doc = RV.predictionReportDocument(report, { provenance: 'fresh_app_run', provenanceNote: payload });
    assert.deepEqual(S.tagAudit(doc.replace('<!doctype html>', '').replace(/<style>[^<]*<\/style>/, '').replace(/<meta[^>]*>/g, '').replace(/<\/?(html|head|body|main|title)[^>]*>/g, ''), []), [], payload);
    assert.match(doc, /^<!doctype html><html lang="en">/);
    assert.match(doc, /Content-Security-Policy/);
    assert.ok(!/<script/i.test(doc) && !/https?:\/\//.test(doc.replace(/<!doctype html>/, '')), 'standalone: no script, no remote asset');
    assert.ok(doc.includes('fresh app run'), 'the export carries its provenance label');
  }
});

/* ---- views with hostile profile data ---- */
function hostileState(payload) {
  const st = FLOW.initialState();
  st.data = { phase: 'ready', source: { kind: 'file', name: payload, size: 10 }, datasetId: 'ds_x', limits: { upload_limit_bytes: 8388608, max_rows: 10000, max_features: 100, max_models_per_job: 5, per_model_timeout_seconds: 60, job_timeout_seconds: 300, queue_wait_seconds: 120, dataset_idle_seconds: 1800, finished_job_seconds: 1800 }, expiresAt: null, error: null,
    profile: { rows: 10, columns: 3, parsing: { format_notes: [payload], read_warnings: [payload], note: payload, input_bytes: 100, malformed_rows_skipped_by_reader: 2 },
      schema_columns: [{ column: payload, dtype: payload, unique: 3, missing: 0, id_like: false }, { column: payload + '2', dtype: 'numeric', unique: 10, missing: 1, id_like: true }, { column: 'label', dtype: 'text', unique: 2, missing: 0, id_like: false }],
      target: { column: 'label', suggestion: { column: 'label', confident: true } },
      classes: { column: 'label', labels: [{ label: payload, count: 7 }, { label: payload + 'N', count: 3 }], distinct_labels: 2, missing_target_rows: 0, suggested_positive: { label: payload + 'N', basis: payload }, labels_truncated: false },
      suggested_exclusions: [{ column: payload + '2', reason: payload }],
      limits: { status: 'exceeded', code: payload, message: payload, violations: [payload], status_if_suggested_exclusions_applied: { status: payload } },
      package_profile: { constant_columns: [payload], empty_columns: [payload], duplicate_columns: [[payload, payload]], duplicate_feature_groups: { groups: 1, rows_in_groups: 2, interpretation: payload }, expanded_feature_estimate: { total: 3 } } } };
  st.samples = { phase: 'ready', items: [{ name: payload + '.csv', label: payload, category: payload, desc: payload }], error: null };
  st.target.column = 'label'; st.target.positive = payload; st.target.negative = payload + 'N'; st.target.roles = { [payload + '2']: 'exclude' };
  st.classes = { column: 'label', labels: st.data.profile.classes.labels, distinct: 2, missing: 0, source: 'service' };
  st.errors = [{ field: 'target-column', message: payload }, { field: payload, message: payload }];
  return st;
}
function fakeFlow(st) { return { stepComplete: () => true, canReach: () => true, getState: () => st }; }
test('hostile text is escaped in every step of the analysis flow', () => {
  for (const payload of PAYLOADS) {
    const st = hostileState(payload);
    for (const step of FLOW.STEPS) {
      st.step = step;
      if (step === 'run') { st.run = Object.assign(FLOW.initialState().run, { phase: 'ended', state: 'failed', reason: { code: payload, message: payload }, notice: payload, completed: [{ model: payload, status: payload }], selected: [payload] }); }
      if (step === 'results') { st.result = { report: hostile(result(), payload), provenance: 'fresh_app_run', state: 'partial' }; }
      const html = V.renderAnalyze(st, fakeFlow(st));
      assert.deepEqual(S.tagAudit(html), [], `${step}: ${payload}`);
      assert.ok(!html.includes('<script') && !html.includes('<img') && !html.includes('<iframe'), `${step}: ${payload}`);
      if (step === 'target' || step === 'data') assert.ok(text(html).includes(payload), `${step} shows the text readably`);
    }
    st.run = Object.assign(FLOW.initialState().run, { phase: 'live', state: 'running', stage: 'fitting', current: payload, selected: [payload], completed: [{ model: payload, status: 'succeeded' }], limits: { model: 60, job: 300 } });
    assert.deepEqual(S.tagAudit(V.renderRunLive(st)), [], payload);
    assert.deepEqual(S.tagAudit(V.errorSummary(st)), [], payload);
  }
});
const BIDI_CHARS = ['' + ch(0x202A) + '', '' + ch(0x202B) + '', '' + ch(0x202C) + '', '' + ch(0x202D) + '', '' + ch(0x202E) + '', '' + ch(0x2066) + '', '' + ch(0x2067) + '', '' + ch(0x2068) + '', '' + ch(0x2069) + '', '' + ch(0x200E) + '', '' + ch(0x200F) + '', '' + ch(0x061C) + ''];
const hasBidi = text => BIDI_CHARS.some(c => text.includes(c));

test('text direction controls become visible markers and never reach the markup', () => {
  assert.equal(U.esc('a' + ch(0x202E) + 'b'), 'a&lt;U+202E&gt;b');
  assert.equal(U.markBidi('x' + ch(0x2066) + 'y' + ch(0x2069) + ''), 'x<U+2066>y<U+2069>');
  BIDI_CHARS.forEach(c => { const out = U.esc('p' + c + 'q'); assert.ok(!hasBidi(out), c.charCodeAt(0).toString(16)); assert.match(out, /^p&lt;U\+[0-9A-F]{4}&gt;q$/); });
  assert.equal(U.esc('plain text, 100% fine'), 'plain text, 100% fine');
  const payload = 'gnp.' + ch(0x202E) + 'txt<' + '' + ch(0x200F) + '' + 'x' + ch(0x061C) + '';
  const html = RV.renderTrainingReport(hostile(result(), payload), { provenance: 'fresh_app_run', idPrefix: 'bd' });
  assert.ok(!hasBidi(html)); assert.ok(html.includes('&lt;U+202E&gt;') && html.includes('&lt;U+200F&gt;') && html.includes('&lt;U+061C&gt;'));
  assert.deepEqual(S.tagAudit(html), []);
  const st = hostileState(payload);
  for (const step of FLOW.STEPS) { st.step = step; if (step === 'results') st.result = { report: hostile(result(), payload), provenance: 'fresh_app_run', state: 'succeeded' }; assert.ok(!hasBidi(V.renderAnalyze(st, fakeFlow(st))), step); }
  assert.ok(!hasBidi(V.renderRecordedInfo({ title: payload, description: payload, data_origin: payload, note: payload })));
  // the standalone document that is built in the browser, and the live-region text built by the flow
  const parsed = M.fromCSV('y,s\nA,.9\nB,.1\nA,.8\nB,.2\n', 'y', ['s']);
  const labels = parsed.labels.map(l => (l === 'A' ? payload : payload + 'N'));
  const report = M.evaluate(labels, { [payload + 'col']: parsed.scores.s }, { positiveLabel: payload, negativeLabel: payload + 'N', scoreKind: 'probability', reviewBudgets: [1], threshold: 0.5, description: payload });
  const doc = RV.predictionReportDocument(report, { provenance: 'fresh_app_run', provenanceNote: payload });
  assert.ok(!hasBidi(doc) && doc.includes('&lt;U+202E&gt;'));
});

test('the replay command quotes only arguments that contain white space', () => {
  const html = V.renderRecordedInfo({ generator: { regenerate_command: 'python x.py', replay_command_arguments: ['python', '-m', 'rare_event_eval', '--seed=7', '--model=logistic', '--target=outcome', 'evaluation/examples/data.csv', '--note=a b'] } });
  assert.ok(html.includes('python -m rare_event_eval --seed=7 --model=logistic --target=outcome evaluation/examples/data.csv &quot;--note=a b&quot;'), html);
  assert.ok(!html.includes('&quot;--seed=7&quot;') && !html.includes('&quot;--model=logistic&quot;') && !html.includes('&quot;python&quot;'));
});

test('every id in the whole document is unique and every label, labelledby and describedby resolves to exactly one element', () => {
  const index = S.readWeb('index.html');
  const body = index.slice(index.indexOf('<body>'));
  const st = hostileState('plain');
  st.errors = [];
  st.models.costs = { fp: '500', fn: '900', review: '' };
  const parts = {};
  for (const step of FLOW.STEPS) {
    st.step = step;
    if (step === 'results') st.result = { report: result(), provenance: 'fresh_app_run', state: 'succeeded' };
    parts[step] = V.renderAnalyze(st, fakeFlow(st));
  }
  const parsed = M.fromCSV(window.ERF_EXAMPLES[0].csv, 'outcome', ['baseline', 'candidate']);
  const predictionReport = RV.renderPredictionReport(M.evaluate(parsed.labels, parsed.scores, { positiveLabel: 'event', negativeLabel: 'clear', scoreKind: 'probability', reviewBudgets: [2], threshold: 0.5 }), { provenance: 'fresh_app_run', idPrefix: 'pr' });
  const extras = [V.renderOverview({ hosted: true, preview: RV.renderPreview(result()) }), V.renderResearch(window.ERF_STUDY), V.renderHow(), predictionReport,
    RV.renderTrainingReport(result(), { provenance: 'recorded_app_run', idPrefix: 'ex' }), V.renderRecordedInfo({ title: 't' }), V.renderExampleChooser([{ id: 'recorded', title: 'r', provenance: 'recorded_app_run', description: 'd' }], 'recorded')];
  for (const step of FLOW.STEPS) {
    const doc = body + extras.join('') + parts[step];       // the Evaluate predictions form (index.html) and this hosted step are in one document
    assert.deepEqual(S.duplicateIds(doc), [], `step ${step}`);
    const ids = new Map();
    for (const m of doc.matchAll(/\sid="([^"]*)"/g)) ids.set(m[1], (ids.get(m[1]) || 0) + 1);
    const bad = [];
    for (const m of doc.matchAll(/<label[^>]*\sfor="([^"]*)"/g)) if (ids.get(m[1]) !== 1) bad.push('label for ' + m[1] + ' -> ' + (ids.get(m[1]) || 0));
    for (const m of doc.matchAll(/aria-(?:labelledby|describedby)="([^"]*)"/g)) m[1].split(/\s+/).forEach(id => { if (id && ids.get(id) !== 1) bad.push('aria ref ' + id + ' -> ' + (ids.get(id) || 0)); });
    assert.deepEqual(bad, [], `step ${step}`);
  }
  assert.ok(parts.models.includes('id="analyze-cost-fp"') && body.includes('id="cost-fp"'), 'the two cost forms have different ids');
});

test('research, how it works and overview render clean HTML with no duplicate ids', () => {
  const study = window.ERF_STUDY;
  for (const html of [V.renderResearch(study), V.renderHow(), V.renderOverview({ hosted: false, preview: RV.renderPreview(result()) }), V.renderOverview({ hosted: true, preview: '' }), V.renderAnalyzeUnavailable()]) {
    assert.deepEqual(S.tagAudit(html, ['img']), []);
    assert.deepEqual(S.duplicateIds(html), []);
  }
  assert.deepEqual(S.tagAudit(V.renderExampleChooser([{ id: 'a"b', title: '<x>', provenance: 'fictional_fixture', description: '<y>' }], 'a"b')), []);
});

test('the notes of the recorded example are escaped and carry the data origin, the caveat and the regenerate command', () => {
  for (const payload of PAYLOADS) {
    const html = V.renderRecordedInfo({ title: payload, description: payload, data_origin: payload, note: payload, generator: { regenerate_command: payload, replay_command_arguments: ['python', '-m', payload, '--x=' + payload] } });
    assert.deepEqual(S.tagAudit(html), [], payload);
    assert.ok(text(html).includes(payload));
  }
  assert.equal(V.renderRecordedInfo(null), '');
  assert.equal(V.renderRecordedInfo('x'), '');
  assert.deepEqual(S.tagAudit(V.renderRecordedInfo({ title: 'T' })), []);
});

test('the recorded example file written by the generator renders cleanly and is labelled', () => {
  const file = require('node:path').join(S.WEB, 'recorded-example.json');
  if (!require('node:fs').existsSync(file)) return;       // the file is produced by evaluation/examples/make_recorded_example.py
  const rec = JSON.parse(require('node:fs').readFileSync(file, 'utf8'));
  assert.equal(rec.provenance, 'recorded app run');
  assert.equal(rec.schema_version, 'training-1.1');
  const html = RV.renderTrainingReport(rec, { provenance: rec.provenance, idPrefix: 'rec' });
  assert.deepEqual(S.tagAudit(html), []);
  assert.deepEqual(S.duplicateIds(html), []);
  assert.ok(html.includes('recorded app run'));
  assert.deepEqual(S.tagAudit(V.renderRecordedInfo(rec.example)), []);
  const t = text(html);
  assert.match(t, /Dataset, positive class and split/);
  assert.match(t, /Models in this run/);
  Object.values(rec.outcomes).forEach(o => assert.ok(t.includes(o.model_record.display_name), o.model_record.display_name));
  const statement = RV.comparisonStatement(RV.trainingViewModel(rec).models);
  assert.ok(['simplest_best', 'simpler_close'].includes(statement.kind), 'the example shows a case where the simpler model is competitive or better: ' + statement.kind);
});

test('a preview card is rendered for a recorded run and for a fictional fixture, each with its label', () => {
  assert.ok(RV.renderPreview(result(), { provenance: 'recorded_app_run' }).includes('recorded app run'));
  const parsed = M.fromCSV(window.ERF_EXAMPLES[0].csv, 'outcome', ['baseline', 'candidate']);
  const report = M.evaluate(parsed.labels, parsed.scores, { positiveLabel: 'event', negativeLabel: 'clear', scoreKind: 'probability', reviewBudgets: [2] });
  const html = RV.renderPreview(report, { provenance: 'fictional_fixture' });
  assert.ok(html.includes('fictional fixture'));
  assert.deepEqual(S.tagAudit(html), []);
});

test('the saved-prediction report keeps the existing semantics and labels the result', () => {
  const parsed = M.fromCSV(window.ERF_EXAMPLES[2].csv, 'outcome', ['baseline']);
  const report = M.evaluate(parsed.labels, parsed.scores, { positiveLabel: 'event', negativeLabel: 'clear', scoreKind: 'probability', reviewBudgets: [2], threshold: 0.5 });
  const t = text(RV.renderPredictionReport(report, { provenance: 'fictional_fixture', cannotCertify: P.CANNOT_CERTIFY }));
  assert.match(t, /fictional fixture/);
  assert.match(t, /tied scores can yield \d+ to \d+ positives found/);
  assert.match(t, /Top-k score ties use original row order/);
  assert.match(t, /cannot establish how these predictions were generated/);
  assert.match(t, /Training leakage Data availability and domain semantics|Training leakage Not checked/);
});

S.run('test_web_rendering.js');
