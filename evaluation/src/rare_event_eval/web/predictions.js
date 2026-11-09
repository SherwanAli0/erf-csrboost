/* Sherwan Ali. Journey C: evaluate saved predictions entirely in the browser. This module has no network code at all;
   it only calls the metrics core in scoring-core.js. tests/test_web_predictions.js runs it with every network entry point
   replaced by a function that fails the test. */
(function (root) {
  'use strict';
  const M = root.RareEventMetrics;
  const CANNOT_CERTIFY = 'This app cannot establish how these predictions were generated, whether the rows were kept apart from training, or whether the file is a valid independent test set. It evaluates the numbers you give it, under the labels and score direction you declare.';
  const MAX_BYTES = 2000000;

  function parseCosts(costs) {
    const fp = String(costs.fp == null ? '' : costs.fp).trim(), fn = String(costs.fn == null ? '' : costs.fn).trim(), rv = String(costs.review == null ? '' : costs.review).trim();
    const errors = [];
    const num = (text, field, label) => { if (text === '') return null; const n = Number(text); if (!Number.isFinite(n) || n < 0) { errors.push({ field, message: `${label} must be a number of zero or more.` }); return null; } return n; };
    const f = num(fp, 'cost-fp', 'Cost per false alarm'), g = num(fn, 'cost-fn', 'Cost per missed positive'), r = num(rv, 'cost-review', 'Cost per flagged case');
    if ((fp === '') !== (fn === '')) errors.push({ field: fp === '' ? 'cost-fp' : 'cost-fn', message: 'Give both error costs, or leave both blank.' });
    if (!errors.length && fp === '' && fn === '' && (r || 0) !== 0) errors.push({ field: 'cost-fp', message: 'A review cost needs both error costs. Give all three, or leave all blank.' });
    if (errors.length) return { value: null, errors };
    if (fp === '' && fn === '') return { value: null, errors: [] };
    return { value: { false_positive_cost: f, false_negative_cost: g, review_cost: r == null ? 0 : r }, errors: [] };
  }

  /* Returns { report } or { errors: [{field, message}] }. Never throws for user mistakes; never touches the network. */
  function evaluateSaved(csv, form) {
    const errors = [];
    if (!csv) errors.push({ field: 'prediction-file', message: 'Load a CSV file or the fictional example first.' });
    if (!form.labelColumn) errors.push({ field: 'prediction-label', message: 'Choose the true-label column.' });
    if (!form.positiveLabel) errors.push({ field: 'prediction-positive', message: 'Enter the exact positive label.' });
    if (!form.negativeLabel) errors.push({ field: 'prediction-negative', message: 'Enter the exact negative label.' });
    if (form.positiveLabel && form.positiveLabel === form.negativeLabel) errors.push({ field: 'prediction-negative', message: 'The negative label must differ from the positive label.' });
    if (!form.scoreColumns || !form.scoreColumns.length) errors.push({ field: 'score-columns', message: 'Choose at least one score column.' });
    if (form.threshold !== '' && form.threshold != null && !Number.isFinite(Number(form.threshold))) errors.push({ field: 'prediction-threshold', message: 'The fixed threshold must be a number.' });
    const budget = Number(form.budget);
    if (form.budget === '' || !Number.isInteger(budget) || budget < 0) errors.push({ field: 'prediction-budget', message: 'The review budget is a whole number of cases, zero or more.' });
    const costs = parseCosts(form.costs || {});
    errors.push(...costs.errors);
    if (errors.length) return { errors };
    try {
      const parsed = M.fromCSV(csv, form.labelColumn, form.scoreColumns);
      const report = M.evaluate(parsed.labels, parsed.scores, { positiveLabel: form.positiveLabel, negativeLabel: form.negativeLabel, scoreKind: form.scoreKind,
        threshold: form.threshold === '' || form.threshold == null ? null : Number(form.threshold), reviewBudgets: [budget], costs: costs.value,
        description: form.description || 'User-supplied predictions. Training provenance has not been verified.' });
      return { report };
    } catch (e) { return { errors: [{ field: 'prediction-file', message: e.message }] }; }
  }
  /* The JSON download: the report as computed, plus the provenance label. No row-level data is added. */
  function reportJson(report, provenanceLabel) {
    return JSON.stringify({ ...report, provenance: provenanceLabel }, null, 2);
  }
  const api = { evaluateSaved, parseCosts, reportJson, CANNOT_CERTIFY, MAX_BYTES };
  root.FYD = root.FYD || {};
  root.FYD.predictions = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
