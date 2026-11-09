/* Sherwan Ali. Evaluate predictions (Journey C): browser only. Every network entry point is replaced by a function that
   fails the test, then the whole evaluation runs. */
'use strict';
const assert = require('node:assert/strict');
const S = require('./web_support.js');
const { P, M, RV } = S.load();
const { test } = S;

const CSV = 'outcome,baseline,candidate\nclear,.02,.03\nclear,.08,.10\nevent,.55,.80\nclear,.62,.35\nclear,.15,.12\nclear,.20,.30\nevent,.35,.65\nclear,.10,.05\nclear,.05,.08\nclear,.30,.20\n';
const FORM = { labelColumn: 'outcome', positiveLabel: 'event', negativeLabel: 'clear', scoreColumns: ['baseline', 'candidate'], scoreKind: 'probability', threshold: '0.5', budget: '2', costs: { fp: '', fn: '', review: '0' } };

function poisoned(fn) {
  const names = ['fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'Image'];
  const saved = {}, hits = [];
  names.forEach(n => { saved[n] = globalThis[n]; globalThis[n] = function () { hits.push(n); throw Error('network call: ' + n); }; });
  const nav = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { value: { sendBeacon() { hits.push('sendBeacon'); throw Error('beacon'); } }, configurable: true });
  try { return { value: fn(), hits }; }
  finally { names.forEach(n => { if (saved[n] === undefined) delete globalThis[n]; else globalThis[n] = saved[n]; }); if (nav) Object.defineProperty(globalThis, 'navigator', nav); else delete globalThis.navigator; }
}

test('evaluating, rendering and exporting predictions makes no network request', () => {
  const { value, hits } = poisoned(() => {
    const out = P.evaluateSaved(CSV, FORM);
    const html = RV.renderPredictionReport(out.report, { provenance: 'fresh_app_run', cannotCertify: P.CANNOT_CERTIFY });
    const doc = RV.predictionReportDocument(out.report, { provenance: 'fresh_app_run' });
    const json = P.reportJson(out.report, 'fresh app run');
    return { out, html, doc, json };
  });
  assert.deepEqual(hits, []);
  assert.ok(value.out.report && !value.out.errors);
});

test('the evaluation module has no network or storage code', () => {
  const src = S.readWeb('predictions.js');
  assert.ok(!/fetch|XMLHttpRequest|sendBeacon|WebSocket|localStorage|sessionStorage|indexedDB|document\.|window\.location/.test(src));
});

test('the report equals the metrics core output (parity is tested separately against Python fixtures)', () => {
  const parsed = M.fromCSV(CSV, 'outcome', ['baseline', 'candidate']);
  const expected = M.evaluate(parsed.labels, parsed.scores, { positiveLabel: 'event', negativeLabel: 'clear', scoreKind: 'probability', threshold: 0.5, reviewBudgets: [2], costs: null, description: 'User-supplied predictions. Training provenance has not been verified.' });
  const got = P.evaluateSaved(CSV, FORM).report;
  assert.deepEqual(got.models, expected.models);
  assert.deepEqual(got.data, expected.data);
  assert.equal(got.checks.training_leakage, 'not_checked');
});

test('every input problem is listed with its field', () => {
  const bad = P.evaluateSaved('', { labelColumn: '', positiveLabel: '', negativeLabel: '', scoreColumns: [], scoreKind: 'probability', threshold: 'x', budget: '-1', costs: { fp: '1', fn: '', review: '' } });
  assert.deepEqual(bad.errors.map(e => e.field).sort(), ['cost-fn', 'prediction-budget', 'prediction-file', 'prediction-label', 'prediction-negative', 'prediction-positive', 'prediction-threshold', 'score-columns'].sort());
  assert.equal(P.evaluateSaved(CSV, { ...FORM, negativeLabel: 'event' }).errors[0].field, 'prediction-negative');
  const wrongLabel = P.evaluateSaved(CSV, { ...FORM, positiveLabel: 'EVENT' });
  assert.match(wrongLabel.errors[0].message, /Every label must match one of the two declared classes exactly/);
  assert.match(P.evaluateSaved('a,b\n1', FORM).errors[0].message, /wrong field count|selected column is missing/);
  assert.match(P.evaluateSaved(CSV.replace('.03', 'abc'), FORM).errors[0].message, /Each score must be numeric/);
});

test('cost assumptions: blank is none, both error costs are required together, review cost defaults to zero', () => {
  assert.equal(P.parseCosts({ fp: '', fn: '', review: '0' }).value, null);
  assert.deepEqual(P.parseCosts({ fp: '1', fn: '10', review: '' }).value, { false_positive_cost: 1, false_negative_cost: 10, review_cost: 0 });
  assert.ok(P.parseCosts({ fp: '1', fn: '', review: '' }).errors.length);
  assert.ok(P.parseCosts({ fp: '', fn: '', review: '3' }).errors.length);
  const out = P.evaluateSaved(CSV, { ...FORM, costs: { fp: '1', fn: '10', review: '0.5' } });
  assert.equal(typeof out.report.models.baseline.threshold_result.assumed_total_cost, 'number');
  assert.match(RV.renderPredictionReport(out.report, { provenance: 'fresh_app_run' }), /assumptions, not measured savings/);
});

test('the download holds aggregates and the provenance label, no per-row data', () => {
  const out = P.evaluateSaved(CSV, FORM);
  const json = P.reportJson(out.report, 'fresh app run');
  const parsed = JSON.parse(json);
  assert.equal(parsed.provenance, 'fresh app run');
  assert.ok(!/NaN|Infinity/.test(json));
  assert.equal(parsed.schema_version, '1.0');
  const scoresSeen = ['.55', '0.55', '.62', '0.62', '.35'].some(v => json.includes(`"${v}"`));
  assert.ok(!scoresSeen);
  assert.ok(!('labels' in parsed) && !('scores' in parsed) && !('rows_data' in parsed));
  const arrays = JSON.stringify(parsed, (k, v) => (Array.isArray(v) && v.length === 10 && v.every(x => typeof x === 'number')) ? '__ROWS__' : v);
  assert.ok(!arrays.includes('__ROWS__'), 'no array with one value per input row');
  const doc = RV.predictionReportDocument(out.report, { provenance: 'fresh_app_run' });
  assert.ok(doc.includes('fresh app run') && doc.includes('lang="en"'));
});

test('the page text says what this mode does and does not establish', () => {
  const C = globalThis.FYD.content;
  assert.match(C.PRIVACY.browser, /No request carries the file or its rows/);
  assert.match(P.CANNOT_CERTIFY, /cannot establish how these predictions were generated/);
  assert.match(P.CANNOT_CERTIFY, /valid independent test set/);
  const html = RV.renderPredictionReport(P.evaluateSaved(CSV, FORM).report, { provenance: 'fresh_app_run', cannotCertify: P.CANNOT_CERTIFY });
  assert.ok(html.includes('cannot establish how these predictions were generated'));
});

S.run('test_web_predictions.js');
