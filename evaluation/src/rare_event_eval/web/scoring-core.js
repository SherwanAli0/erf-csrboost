/* Sherwan Ali. Browser counterpart of the saved-prediction report contract. */
(function (root) {
  'use strict';
  const ratio = (a, b) => b ? a / b : null;
  function parseCSV(text) {
    text = text.replace(/^\uFEFF/, '');
    const rows = [], row = [];
    let field = '', quoted = false, closed = false, started = false;
    function pushField() { row.push(field); field = ''; closed = false; started = false; }
    function pushRow() { pushField(); rows.push(row.splice(0)); }
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else { quoted = false; closed = true; }
        } else field += c;
      } else if (c === '"') {
        if (started || closed) throw Error('Malformed CSV quoting.');
        quoted = true; started = true;
      } else if (c === ',') pushField();
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        pushRow();
      } else {
        if (closed) throw Error('Unexpected text after a quoted CSV field.');
        field += c; started = true;
      }
    }
    if (quoted) throw Error('Unclosed CSV quote.');
    if (started || closed || field || row.length) pushRow();
    if (!rows.length) throw Error('CSV is empty.');
    const headers = rows.shift();
    if (headers.some(x => !x.trim()) || new Set(headers).size !== headers.length) throw Error('Headers must be nonempty and unique.');
    if (rows.some(r => r.length !== headers.length)) throw Error('A CSV row has the wrong field count. No rows were dropped.');
    if (rows.length > 20000) throw Error('Browser evaluation supports up to 20,000 rows. Use the Python CLI for larger files.');
    return { headers, rows };
  }
  function numeric(value) {
    if (typeof value === 'number') return value;
    const text = String(value).trim();
    if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(text)) throw Error('Each score must be numeric.');
    return Number(text);
  }
  function fromCSV(csv, labelColumn, scoreColumns) {
    const { headers, rows } = parseCSV(csv);
    if (!scoreColumns.length || new Set(scoreColumns).size !== scoreColumns.length || scoreColumns.includes(labelColumn)) throw Error('Choose distinct score columns and a separate label column.');
    if ([labelColumn, ...scoreColumns].some(c => !headers.includes(c))) throw Error('A selected column is missing.');
    const scores = Object.create(null);
    scoreColumns.forEach(name => { scores[name] = rows.map(r => numeric(r[headers.indexOf(name)])); });
    return { labels: rows.map(r => r[headers.indexOf(labelColumn)]), scores };
  }
  function decision(y, selected, costs) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    y.forEach((v, i) => { if (selected[i]) v ? tp++ : fp++; else v ? fn++ : tn++; });
    const recall = ratio(tp, tp + fn), specificity = ratio(tn, tn + fp);
    const result = { true_positives: tp, false_positives: fp, false_negatives: fn, true_negatives: tn,
      flagged: tp + fp, flagged_fraction: (tp + fp) / y.length, precision: ratio(tp, tp + fp), recall,
      specificity, false_positive_rate: ratio(fp, tn + fp), f1: ratio(2 * tp, 2 * tp + fp + fn),
      accuracy: (tp + tn) / y.length, balanced_accuracy: recall !== null && specificity !== null ? (recall + specificity) / 2 : null,
      assumed_total_cost: costs ? fp * costs.false_positive_cost + fn * costs.false_negative_cost + (tp + fp) * costs.review_cost : null };
    if (costs && !Number.isFinite(result.assumed_total_cost)) throw Error('Calculated cost exceeds the numeric range.');
    return result;
  }
  function evaluate(labels, modelScores, options) {
    const { positiveLabel, negativeLabel, scoreKind } = options;
    if (typeof positiveLabel !== 'string' || typeof negativeLabel !== 'string' || !positiveLabel.trim() || !negativeLabel.trim() || positiveLabel === negativeLabel) throw Error('Choose distinct, nonempty positive and negative labels.');
    if (!labels.length || labels.some(v => v !== positiveLabel && v !== negativeLabel)) throw Error('Every label must match one of the two declared classes exactly.');
    if (!['probability', 'ranking'].includes(scoreKind)) throw Error('Choose probability or ranking scores.');
    const y = labels.map(v => Number(v === positiveLabel)), n = y.length, positives = y.reduce((a,b) => a+b, 0);
    const budgets = options.reviewBudgets || [];
    if (new Set(budgets).size !== budgets.length || budgets.some(k => !Number.isInteger(k) || k < 0 || k > n)) throw Error('Budgets must be unique integer counts from zero to the number of rows.');
    const threshold = options.threshold == null ? (scoreKind === 'probability' ? .5 : null) : options.threshold;
    if (threshold !== null && (!Number.isFinite(threshold) || scoreKind === 'probability' && (threshold < 0 || threshold > 1))) throw Error('Invalid fixed threshold.');
    const costs = options.costs || null;
    if (costs && ['false_positive_cost', 'false_negative_cost', 'review_cost'].some(k => typeof costs[k] !== 'number' || !Number.isFinite(costs[k]) || costs[k] < 0)) throw Error('All cost assumptions must be finite and nonnegative.');
    const models = Object.create(null);
    if (!Object.keys(modelScores).length) throw Error('At least one score column is required.');
    Object.entries(modelScores).forEach(([name, scores]) => {
      if (!name.trim() || scores.length !== n || scores.some(s => typeof s !== 'number' || !Number.isFinite(s))) throw Error('Every model must have one finite score for each row.');
      if (scoreKind === 'probability' && scores.some(s => s < 0 || s > 1)) throw Error('Probabilities must be between zero and one.');
      const order = Array.from({ length: n }, (_, i) => i).sort((a,b) => scores[b] - scores[a] || a-b);
      let tp = 0, fp = 0, ap = 0, auc = 0;
      if (positives && positives < n) {
        for (let start = 0; start < n;) {
          let end = start, groupPositives = 0;
          while (end < n && scores[order[end]] === scores[order[start]]) { groupPositives += y[order[end]]; end++; }
          const nextTP = tp + groupPositives, nextFP = fp + end - start - groupPositives;
          ap += nextTP / end * groupPositives / positives;
          auc += (nextFP - fp) / (n - positives) * (tp + nextTP) / (2 * positives);
          tp = nextTP; fp = nextFP; start = end;
        }
      } else { ap = null; auc = null; }
      let brier = null, loss = null, calibration = null;
      if (scoreKind === 'probability') {
        brier = scores.reduce((sum, s, i) => sum + (s - y[i]) ** 2, 0) / n;
        loss = scores.reduce((sum, s, i) => { const p = Math.max(Number.EPSILON, Math.min(1-Number.EPSILON, s)); return sum - (y[i] ? Math.log(p) : Math.log1p(-p)); }, 0) / n;
        const bins = Array.from({ length: 10 }, (_, i) => ({ lower: i/10, upper: (i+1)/10, upper_inclusive: i === 9, count: 0, sum: 0, positive: 0 }));
        scores.forEach((s, i) => { const bin = bins[Math.min(9, Math.floor(s * 10))]; bin.count++; bin.sum += s; bin.positive += y[i]; });
        calibration = { strategy: 'equal_width', bins: bins.map(b => ({ lower:b.lower, upper:b.upper, upper_inclusive:b.upper_inclusive, count:b.count,
          mean_score:b.count ? b.sum/b.count : null, observed_positive_fraction:b.count ? b.positive/b.count : null })), interpretation:'Descriptive reliability bins; calibration is not certified.' };
      }
      const budgetResults = budgets.map(k => {
        const selected = Array(n).fill(false); order.slice(0,k).forEach(i => { selected[i] = true; });
        const result = { budget:k, ...decision(y,selected,costs), boundary_tie:null };
        if (k > 0 && k < n && scores[order[k-1]] === scores[order[k]]) {
          const boundary = scores[order[k-1]], tied = order.filter(i => scores[i] === boundary), above = order.filter(i => scores[i] > boundary);
          const slots = k - above.length, p = tied.reduce((s,i) => s+y[i],0), existing = above.reduce((s,i) => s+y[i],0);
          const low = existing + Math.max(0, slots - (tied.length-p)), high = existing + Math.min(slots,p);
          result.boundary_tie = { score:boundary, tied_rows:tied.length, available_slots:slots, true_positives_min:low, true_positives_max:high,
            precision_min:low/k, precision_max:high/k, recall_min:ratio(low,positives), recall_max:ratio(high,positives), interpretation:'Tie-allocation bounds, not confidence intervals.' };
        }
        return result;
      });
      const warnings = [];
      if (!positives || positives === n) warnings.push('Only one class is present. AP and ROC AUC are unavailable.');
      if (positives < 20) warnings.push('Few positive rows: estimates can be unstable. This is descriptive, not a validity cutoff.');
      if (budgetResults.some(row => row.boundary_tie)) warnings.push('A budget cuts through a score tie. Inspect the possible outcome bounds.');
      models[name] = { metrics:{ average_precision:ap, roc_auc:auc, brier_score:brier, log_loss:loss },
        threshold_result:threshold === null ? null : { threshold, ...decision(y,scores.map(s => s >= threshold),costs) },
        review_budget_results:budgetResults, calibration, warnings };
    });
    return { schema_version:'1.0', tool:{ name:'rare-event-eval-browser', version:'0.3.0', engine:'Browser JavaScript; numerical parity checked against the Python core' },
      created_utc:new Date().toISOString(), task:{ kind:'binary_classification', positive_label:positiveLabel, negative_label:negativeLabel, score_kind:scoreKind, evaluation_role:'unspecified', role_verified:false },
      data:{ rows:n, positive_rows:positives, negative_rows:n-positives, positive_prevalence:positives/n, rows_dropped:0, description:options.description || 'User-supplied saved predictions.', description_verified:false },
      policies:{ threshold, threshold_rule:'score >= threshold', review_budgets:budgets, tie_break:'input row order', cost_assumptions:costs,
        selection:'Fixed policies. Changing them after inspecting evaluation outcomes is exploratory.' },
      checks:{ labels_and_scores:'validated', training_leakage:'not_checked', train_test_overlap:'not_checked', temporal_or_group_independence:'not_checked', uncertainty:'not_estimated' },
      models, limitations:[ 'Saved predictions cannot certify leakage-free training or independent evaluation.', 'Class orientation and row alignment must be supplied correctly.',
        'Threshold and budget exploration on test labels does not select an independently validated final policy.', 'No confidence interval or automatic winning model is produced.',
        'Probability scores and reliability bins do not certify calibration.', 'Costs are hypothetical assumptions, not measured savings.', 'Reports exclude raw rows but aggregate results may still be sensitive.' ] };
  }
  const api = { parseCSV, fromCSV, evaluate };
  root.RareEventMetrics = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
