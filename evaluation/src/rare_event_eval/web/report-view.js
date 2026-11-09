/* Sherwan Ali. Pure renderers: a training-1.1 comparison report (hosted run or recorded run) and a saved-prediction
   report (schema 1.0) become HTML strings. Every value that came from data, a report or a server is escaped.
   No DOM, no network. */
(function (root) {
  'use strict';
  const U = root.FYD.util, C = root.FYD.content, esc = U.esc;
  const TIE_AP = 0.01;          // display convention for the word "near-tied"; not a statistical test
  const FEW_POSITIVES = 20;     // matches the warning threshold of the browser metrics core
  const COST_RANK = { light: 0, moderate: 1, heavy: 2 };

  function provenanceKey(value, fallback) {
    if (typeof value === 'string') {
      if (C.PROVENANCE[value]) return value;
      const hit = Object.keys(C.PROVENANCE).find(key => C.PROVENANCE[key].label === value);
      if (hit) return hit;
    }
    return fallback || 'fresh_app_run';
  }
  function badge(key, note) {
    const p = C.PROVENANCE[key];
    return `<p class="provenance provenance-${esc(key.replace(/_/g, '-'))}"><span class="badge badge-provenance"><span class="badge-mark" aria-hidden="true">${esc(p.mark)}</span><span class="badge-label"><span class="visually-hidden">Result provenance: </span>${esc(p.label)}</span></span> <span class="provenance-text">${esc(note || p.explain)}</span></p>`;
  }
  function term(key, summaryText) {
    const g = C.GLOSSARY[key];
    return `<details class="term"><summary>${esc(summaryText || ('What is ' + g.term.toLowerCase() + '?'))}</summary><p>${esc(g.text)}</p></details>`;
  }
  function makeIds(prefix) { let n = 0; return () => `${prefix}-${++n}`; }
  /* A table inside a labelled, keyboard-scrollable region. cells: string | number | null | {html}. */
  function table(ids, caption, headers, rows, options = {}) {
    const id = ids(), cell = v => (v && typeof v === 'object' && 'html' in v) ? v.html : esc(v == null ? U.UNAVAILABLE : v);
    const head = headers.map(h => `<th scope="col"${h.num ? ' class="num"' : ''}>${esc(h.text)}</th>`).join('');
    const body = rows.map(r => `<tr>${r.map((v, i) => {
      const numeric = headers[i] && headers[i].num;
      return i === 0 && options.rowHeader !== false ? `<th scope="row">${cell(v)}</th>` : `<td${numeric ? ' class="num"' : ''}>${cell(v)}</td>`;
    }).join('')}</tr>`).join('');
    return `<div class="table-scroll" role="region" aria-labelledby="${id}" tabindex="0"><table><caption id="${id}">${esc(caption)}</caption><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }
  function stat(label, value, sub) {
    return `<div class="stat"><span class="stat-label">${esc(label)}</span><strong class="stat-value">${esc(value)}</strong>${sub ? `<span class="stat-sub">${esc(sub)}</span>` : ''}</div>`;
  }
  /* A bar list. Each row has its name, a bar, and the printed value, so colour is never the only carrier. */
  function bars(ids, title, items, options = {}) {
    const id = ids(), base = options.baseline;
    const rows = items.map(item => {
      const v = typeof item.value === 'number' && Number.isFinite(item.value) ? Math.max(0, Math.min(1, item.value)) : null;
      const label = `${item.label}: ${v == null ? U.UNAVAILABLE : U.fixed(item.value, 3)}`;
      // The marker is taller than the bar and has a light halo, so it stays visible over the filled part of a bar.
      const bx = base ? Math.max(0.6, Math.min(98.6, base.value * 100)) : 0;
      const baseRect = base ? `<rect class="bar-halo" x="${(bx - 0.6).toFixed(2)}" y="0" width="2" height="14"></rect><rect class="bar-base" x="${bx.toFixed(2)}" y="0" width="0.8" height="14"></rect>` : '';
      return `<li class="bar-row"><span class="bar-label">${esc(item.label)}${item.tag ? `<span class="bar-tag">${esc(item.tag)}</span>` : ''}</span><svg class="bar" viewBox="0 0 100 14" preserveAspectRatio="none" role="img" aria-label="${esc(label)}"><rect class="bar-track" x="0" y="3" width="100" height="8"></rect><rect class="bar-fill" x="0" y="3" width="${v == null ? 0 : (v * 100).toFixed(2)}" height="8"></rect>${baseRect}</svg><span class="bar-value">${v == null ? U.UNAVAILABLE : U.fixed(item.value, 3)}</span></li>`;
    }).join('');
    const legend = base ? `<p class="chart-key"><span class="key-swatch key-fill" aria-hidden="true"></span> model value <span class="key-swatch key-base" aria-hidden="true"></span> ${esc(base.label)}</p>` : '';
    return `<figure class="chart" aria-labelledby="${id}"><figcaption id="${id}">${esc(title)}</figcaption><ul class="bars">${rows}</ul>${legend}${options.summary ? `<p class="chart-summary">${esc(options.summary)}</p>` : ''}</figure>`;
  }

  function capacityText(record) {
    const est = record && record.params && record.params.estimator;
    if (!est) return U.UNAVAILABLE;
    if (est.n_bags && est.trees_per_bag) return `${est.n_bags} bags x ${est.trees_per_bag} trees (${U.int(est.n_bags * est.trees_per_bag)} trees)`;
    if (est.n_estimators) return /Boost/.test(String(est.class)) ? `${est.n_estimators} boosting rounds` : `${est.n_estimators} trees`;
    if (est.max_iter && /Boosting/.test(String(est.class))) return `${est.max_iter} iterations`;
    if (/Logistic/.test(String(est.class))) return 'linear model';
    return U.UNAVAILABLE;
  }
  function objectiveText(sel) {
    if (!sel) return U.UNAVAILABLE;
    const text = { maximize_f1: 'maximizes F1 on the validation partition', minimize_assumed_cost: 'minimizes the assumed cost on the validation partition' }[sel.objective];
    return text || String(sel.objective == null ? U.UNAVAILABLE : sel.objective);
  }
  function fewPositives(count) { return typeof count === 'number' && count < FEW_POSITIVES; }

  /* ---- view model for training-1.1 ---- */
  function trainingViewModel(result) {
    if (!result || typeof result !== 'object' || !result.outcomes || typeof result.outcomes !== 'object') throw Error('Not a training report.');
    const entries = Object.entries(result.outcomes);
    const records = result.model_records || {};
    const models = [], failures = [];
    entries.forEach(([id, o], index) => {
      const record = o.model_record || records[id] || {};
      const base = { id, name: record.display_name || id, record, order: index, cost: record.cost_class || 'moderate' };
      if (o.status === 'success' && o.report && o.report.models && o.report.models[id]) {
        const m = o.report.models[id];
        models.push({ ...base, metrics: m.metrics || {}, threshold: m.threshold_result || null, budgets: m.review_budget_results || [],
          calibration: m.calibration || null, warnings: [...(m.warnings || []), ...((o.execution && o.execution.warnings) || [])],
          execution: o.execution || {}, selection: (o.execution && o.execution.threshold_selection) || null,
          wall: o.worker_wall_seconds, test: o.report.data || {}, policies: o.report.policies || {} });
      } else {
        failures.push({ ...base, status: o.status, code: o.reason_code || (o.status === 'timeout' ? 'MODEL_TIMEOUT' : null),
          message: o.error || null, timeout: o.timeout_seconds });
      }
    });
    const first = models[0] || null;
    const mapping = result.label_mapping || {};
    return { result, models, failures, first, mapping, counts: mapping.counts || {},
      split: result.split || {}, parsing: result.input_parsing || {}, excluded: result.excluded_columns || [],
      predictors: result.predictors || result.feature_names || [], kinds: result.predictor_kinds || null,
      seed: result.seed, hash: result.input_sha256 || null, created: first ? ((result.outcomes[first.id].report || {}).created_utc || null) : null,
      tool: first ? ((result.outcomes[first.id].report || {}).tool || null) : null };
  }

  /* The sentence about whether a simpler model is competitive. kind: none | single | simplest_best | simpler_close | heavier_clear. */
  function comparisonStatement(models) {
    const scored = models.filter(m => typeof m.metrics.average_precision === 'number');
    if (models.length === 0) return { kind: 'none', text: 'No model produced a result, so there is nothing to compare.' };
    if (models.length === 1 || scored.length < 2) return { kind: 'single', text: 'Only one model produced a result with an average precision, so this report contains no comparison.' };
    const best = scored.reduce((a, b) => (b.metrics.average_precision > a.metrics.average_precision + 1e-12 || (Math.abs(b.metrics.average_precision - a.metrics.average_precision) <= 1e-12 && COST_RANK[b.cost] < COST_RANK[a.cost])) ? b : a);
    const simpler = scored.filter(m => (COST_RANK[m.cost] ?? 1) < (COST_RANK[best.cost] ?? 1));
    const bestAp = best.metrics.average_precision;
    if (!simpler.length) {
      const heavier = scored.some(m => (COST_RANK[m.cost] ?? 1) > (COST_RANK[best.cost] ?? 1));
      return { kind: 'simplest_best', best: best.id, text: `${best.name} has the highest average precision (${U.fixed(bestAp)}) on this test partition${heavier ? ', and no heavier model improved on it' : ''}. With one split this is an estimate, not a ranking that is certain to hold on other data.` };
    }
    const close = simpler.reduce((a, b) => b.metrics.average_precision > a.metrics.average_precision ? b : a);
    const gap = bestAp - close.metrics.average_precision;
    const times = (typeof best.execution.fit_seconds === 'number' && typeof close.execution.fit_seconds === 'number' && close.execution.fit_seconds > 0)
      ? ` ${best.name} took ${U.seconds(best.execution.fit_seconds)} to fit and ${close.name} took ${U.seconds(close.execution.fit_seconds)}.` : '';
    if (gap < TIE_AP) {
      return { kind: 'simpler_close', best: best.id, simple: close.id, gap, text: `The simpler model ${close.name} is competitive. Its average precision (${U.fixed(close.metrics.average_precision)}) is within ${U.fixed(gap)} of the highest, ${best.name} (${U.fixed(bestAp)}). With one split and no uncertainty interval, a gap this small can change under a different split, so cost, time and interpretability can reasonably decide.${times} "Within ${U.fixed(TIE_AP, 2)}" is a display rule on this page, not a statistical test.` };
    }
    return { kind: 'heavier_clear', best: best.id, simple: close.id, gap, text: `${best.name} has higher average precision (${U.fixed(bestAp)}) than the simpler model ${close.name} (${U.fixed(close.metrics.average_precision)}), a gap of ${U.fixed(gap)} on this test partition.${times} This is one split without an uncertainty interval, so confirm it on other data before relying on it.` };
  }

  function sectionHead(level, title, intro) {
    return `<h${level}>${esc(title)}</h${level}>${intro ? `<p class="section-intro">${esc(intro)}</p>` : ''}`;
  }

  function renderDatasetSection(vm, ids, hl) {
    const c = vm.counts, m = vm.mapping, test = vm.first ? vm.first.test : {};
    const parts = vm.split.achieved && vm.split.achieved.partitions;
    const req = vm.split.requested || {};
    const negatives = Array.isArray(m.negative_labels) ? m.negative_labels : (m.negative_label != null ? [m.negative_label] : []);
    const modeText = m.mode === 'one_vs_rest' ? 'One label against all other labels' : 'Two labels';
    const prevalence = typeof c.positive_rows === 'number' && c.effective_rows ? c.positive_rows / c.effective_rows : null;
    const tiles = `<div class="stats">${stat('Rows used', U.int(c.effective_rows), c.input_rows != null && c.effective_rows !== c.input_rows ? `${U.int(c.input_rows)} in the file` : (c.input_rows != null ? 'all rows in the file' : ''))}${stat('Positive cases', U.int(c.positive_rows), U.pct(prevalence) + ' of rows used')}${stat('Test partition', U.int(test.rows), `${U.int(test.positive_rows)} positive`)}</div>`;
    const dl = `<dl class="facts"><div><dt>Positive class</dt><dd>${esc(m.positive_label)} <span class="muted">(chosen explicitly, never inferred)</span></dd></div><div><dt>Negative class</dt><dd>${m.mode === 'one_vs_rest' ? 'All other labels: ' : ''}${negatives.length ? U.listText(negatives) : U.UNAVAILABLE}</dd></div><div><dt>Label mode</dt><dd>${esc(modeText)}</dd></div><div><dt>Missing-target policy</dt><dd>${esc(m.missing_policy === 'error' ? 'stop with an error' : m.missing_policy === 'drop_reported' ? 'exclude and list' : (m.missing_policy == null ? U.UNAVAILABLE : m.missing_policy))}${c.missing_target_rows ? `; ${U.int(c.missing_target_rows)} rows without a target were excluded and listed` : ''}</dd></div><div><dt>Seed</dt><dd>${esc(vm.seed == null ? U.UNAVAILABLE : vm.seed)}</dd></div>${vm.hash ? `<div><dt>Input SHA-256</dt><dd class="mono">${esc(vm.hash)}</dd></div>` : ''}</dl>`;
    const labels = Array.isArray(m.original_labels) ? table(ids, 'Original labels in the target column', [{ text: 'Label' }, { text: 'Rows', num: true }, { text: 'Used as' }],
      m.original_labels.map(l => [l.label, U.int(l.count), l.internal === 1 ? 'positive' : 'negative'])) : '';
    let splitHtml;
    if (parts) {
      const rows = ['train', 'validation', 'test'].filter(n => parts[n]).map(n => [n, U.int(parts[n].rows), U.int(parts[n].positive_rows), U.int(parts[n].negative_rows), U.pct(parts[n].fraction_of_rows)]);
      const frac = req.target_fractions ? ` Target shares were ${['train', 'validation', 'test'].map(n => `${n} ${U.pct(req.target_fractions[n], 0)}`).join(', ')}; achieved counts are exact.` : '';
      splitHtml = `<p>Requested split: <strong>${esc(req.mode)}</strong>, seed ${esc(req.seed == null ? U.UNAVAILABLE : req.seed)}.${esc(frac)} Achieved: <strong>${esc(vm.split.achieved.mode)}</strong>; the same partitions were used for every model: <strong>${vm.split.achieved.identical_across_models ? 'yes' : 'no'}</strong>; fallback split used: <strong>${vm.split.achieved.fallback_used ? 'yes' : 'no'}</strong>.</p>` +
        table(ids, 'Achieved partitions', [{ text: 'Partition' }, { text: 'Rows', num: true }, { text: 'Positive', num: true }, { text: 'Negative', num: true }, { text: 'Share of rows', num: true }], rows) + term('partitions', 'What are train, validation and test?');
    } else {
      splitHtml = `<p class="notice">No achieved split is recorded because no model completed. ${esc(vm.split.note || '')}</p>`;
    }
    const ov = vm.first && vm.first.execution.split;
    let overlap = '';
    if (ov) {
      overlap = `<p>Repeated feature patterns: ${U.int(ov.duplicate_feature_groups)} groups covering ${U.int(ov.rows_in_duplicate_groups)} rows; ${U.int(ov.conflicting_feature_patterns)} patterns carry both classes; ${U.int(ov.test_rows_matching_training_features)} test rows have a feature twin in the train partition. ${esc(ov.interpretation || '')}</p>${term('overlap', 'How should I read repeated feature patterns?')}`;
    }
    const excl = vm.excluded.length ? table(ids, 'Columns that are not predictors', [{ text: 'Column' }, { text: 'Role' }, { text: 'Why' }], vm.excluded.map(e => [e.column, e.role, e.reason || ''])) : '<p>No column was excluded by role.</p>';
    const preds = vm.predictors.length ? `<details class="more"><summary>${U.int(vm.predictors.length)} predictors used</summary><p class="predictor-list">${vm.predictors.map(p => `<span class="pill">${esc(p)}${vm.kinds && vm.kinds[p] ? ` <span class="muted">${esc(vm.kinds[p])}</span>` : ''}</span>`).join(' ')}</p></details>` : '';
    const pd = vm.parsing.details || {};
    const notes = [].concat(pd.format_notes || [], pd.read_warnings || []);
    const parsing = `<details class="more"><summary>How the file was read</summary><p>Parser: ${esc(pd.parser || vm.parsing.parser || U.UNAVAILABLE)}. Rows read ${U.int(vm.parsing.rows_input)}, used ${U.int(vm.parsing.rows_used)}, excluded ${U.int(vm.parsing.rows_excluded)}.</p>${notes.length ? `<ul>${notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>` : '<p>The reader recorded no format notes or warnings.</p>'}${pd.note ? `<p class="muted">${esc(pd.note)}</p>` : ''}</details>`;
    return `<section class="report-section" aria-labelledby="${ids.prefix}-dataset">${sectionHead(hl, 'Dataset, positive class and split').replace('<h' + hl + '>', `<h${hl} id="${ids.prefix}-dataset">`)}${tiles}${dl}${labels}<h${hl + 1}>How the data was split</h${hl + 1}>${splitHtml}${overlap}<h${hl + 1}>Columns</h${hl + 1}>${excl}${preds}${parsing}</section>`;
  }

  function renderModelsSection(vm, ids, hl) {
    const rows = [];
    vm.models.forEach(m => rows.push([m.name, { html: '<span class="status status-ok">Result produced</span>' }, 'none', m.record.recipe_id || U.UNAVAILABLE, capacityText(m.record), m.cost]));
    vm.failures.forEach(f => {
      const reason = f.code ? `${f.code}` : (f.status || U.UNAVAILABLE);
      rows.push([f.name, { html: `<span class="status status-bad">${esc(f.status === 'timeout' ? 'Timed out' : 'No result')}</span>` }, reason, f.record.recipe_id || U.UNAVAILABLE, capacityText(f.record), f.cost]);
    });
    const failuresList = vm.failures.length ? `<div class="callout callout-warn" role="group" aria-label="Models without a result"><h${hl + 1}>Models without a result</h${hl + 1}><ul>${vm.failures.map(f => {
      const plain = f.code && C.REASONS[f.code] ? C.REASONS[f.code] : 'The model did not produce a result.';
      return `<li><strong>${esc(f.name)}</strong> (${esc(f.code || f.status)}): ${esc(plain)}${f.message ? ` Message: ${esc(f.message)}` : ''}${typeof f.timeout === 'number' ? ` Time limit: ${esc(U.seconds(f.timeout))}.` : ''}</li>`;
    }).join('')}</ul><p>These models are part of the comparison. They were not dropped, and no other recipe was substituted.</p></div>` : '';
    const state = vm.failures.length ? `<p class="notice">Partial result: ${U.int(vm.models.length)} of ${U.int(vm.models.length + vm.failures.length)} models produced a result.</p>` : '';
    const header = rows.length ? table(ids, 'Models in this run', [{ text: 'Model' }, { text: 'Outcome' }, { text: 'Reason code' }, { text: 'Recipe id' }, { text: 'Capacity' }, { text: 'Cost class' }], rows) + term('cost_class', 'What is the cost class?') : '';
    return `<section class="report-section">${sectionHead(hl, 'Models in this run')}${state}${header}${failuresList}</section>`;
  }

  function renderRanking(vm, ids, hl) {
    if (!vm.models.length) return '';
    const withAp = vm.models.filter(m => typeof m.metrics.average_precision === 'number');
    const bestAp = withAp.length ? Math.max(...withAp.map(m => m.metrics.average_precision)) : null;
    const ref = withAp.length ? withAp.reduce((a, b) => ((COST_RANK[b.cost] ?? 1) < (COST_RANK[a.cost] ?? 1)) ? b : a) : null;
    const prevalence = vm.first.test.positive_prevalence;
    const tag = m => (bestAp != null && typeof m.metrics.average_precision === 'number')
      ? (m.metrics.average_precision === bestAp ? 'highest in this run' : (bestAp - m.metrics.average_precision < TIE_AP ? 'near-tied with the highest' : '')) : '';
    const apChart = bars(ids, 'Average precision on the test partition', vm.models.map(m => ({ label: m.name, value: m.metrics.average_precision, tag: tag(m) })),
      { baseline: { value: typeof prevalence === 'number' ? prevalence : 0, label: `no-skill baseline (positive share ${U.pct(prevalence)})` },
        summary: withAp.length ? `Average precision ranges from ${U.fixed(Math.min(...withAp.map(m => m.metrics.average_precision)))} to ${U.fixed(bestAp)} across ${U.int(withAp.length)} models. The no-skill baseline is ${U.fixed(prevalence)}.` : 'No average precision is available.' });
    const aucs = vm.models.filter(m => typeof m.metrics.roc_auc === 'number');
    const aucChart = aucs.length ? bars(ids, 'ROC AUC on the test partition', vm.models.map(m => ({ label: m.name, value: m.metrics.roc_auc })), { baseline: { value: 0.5, label: 'no-skill baseline (0.5)' },
      summary: `ROC AUC ranges from ${U.fixed(Math.min(...aucs.map(m => m.metrics.roc_auc)))} to ${U.fixed(Math.max(...aucs.map(m => m.metrics.roc_auc)))}.` }) : '';
    const rows = vm.models.map(m => [m.name, U.fixed(m.metrics.average_precision), ref && m !== ref && typeof m.metrics.average_precision === 'number' ? U.signed(m.metrics.average_precision - ref.metrics.average_precision) : (m === ref ? 'reference' : U.UNAVAILABLE),
      U.fixed(m.metrics.roc_auc), ref && m !== ref && typeof m.metrics.roc_auc === 'number' && typeof ref.metrics.roc_auc === 'number' ? U.signed(m.metrics.roc_auc - ref.metrics.roc_auc) : (m === ref ? 'reference' : U.UNAVAILABLE)]);
    const stmt = comparisonStatement(vm.models);
    const caution = fewPositives(vm.first.test.positive_rows) ? `<p class="notice">The test partition has only ${U.int(vm.first.test.positive_rows)} positive cases, so these estimates can be unstable. This is descriptive, not a validity cutoff.</p>` : '';
    return `<section class="report-section">${sectionHead(hl, 'Ranking quality', 'How well the scores put positive cases first, on the same held-out test cases for every model. No threshold is involved.')}<p class="comparison-statement" role="note">${esc(stmt.text)}</p>${caution}${apChart}${aucChart}${table(ids, ref ? `Ranking metrics and differences from the reference model, ${ref.name} (the lightest recipe in this run)` : 'Ranking metrics', [{ text: 'Model' }, { text: 'AP', num: true }, { text: 'AP difference from reference', num: true }, { text: 'ROC AUC', num: true }, { text: 'AUC difference from reference', num: true }], rows)}<p class="table-note">Differences are signed: a negative value means lower than the reference. There is no significance test or interval, and a model is never named the winner automatically.</p>${term('ap')}${term('roc_auc')}</section>`;
  }

  function renderThresholds(vm, ids, hl) {
    const withT = vm.models.filter(m => m.threshold);
    if (!withT.length) return '';
    const costed = withT.some(m => typeof m.threshold.assumed_total_cost === 'number');
    const headers = [{ text: 'Model' }, { text: 'Threshold (from validation)', num: true }, { text: 'Flagged', num: true }, { text: 'Found', num: true }, { text: 'False alarms', num: true }, { text: 'Missed', num: true }, { text: 'Precision', num: true }, { text: 'Recall', num: true }, { text: 'F1', num: true }];
    if (costed) headers.push({ text: 'Assumed cost', num: true });
    const rows = withT.map(m => { const t = m.threshold; const r = [m.name, U.fixed(t.threshold, 2), U.int(t.flagged), U.int(t.true_positives), U.int(t.false_positives), U.int(t.false_negatives), U.pct(t.precision), U.pct(t.recall), U.fixed(t.f1)]; if (costed) r.push(typeof t.assumed_total_cost === 'number' ? U.num(t.assumed_total_cost, 2) : U.UNAVAILABLE); return r; });
    const sel = withT.map(m => [m.name, objectiveText(m.selection), m.selection ? U.int(m.selection.validation_rows) : U.UNAVAILABLE, m.selection ? (m.selection.test_labels_used === false ? 'no' : 'not recorded') : U.UNAVAILABLE]);
    const budgetRows = [], ties = [];
    withT.forEach(m => m.budgets.forEach(b => {
      const share = m.test.rows ? b.budget / m.test.rows : null;
      budgetRows.push([m.name, `Top ${U.int(b.budget)}`, U.pct(share, 0), U.int(b.flagged), U.int(b.true_positives), U.int(b.false_positives), U.pct(b.precision), U.pct(b.recall)]);
      if (b.boundary_tie) ties.push(`${m.name}, top ${U.int(b.budget)}: tied scores can yield ${U.int(b.boundary_tie.true_positives_min)} to ${U.int(b.boundary_tie.true_positives_max)} positive cases found, depending on which tied cases are chosen. These are allocation bounds, not confidence intervals.`);
    }));
    const costNote = costed ? `<p class="notice">Assumed cost uses the numbers you supplied: false positives times the false-positive cost, plus false negatives times the false-negative cost, plus flagged cases times the review cost. These are assumptions, not measured savings.</p>` : '<p class="table-note">No cost assumptions were supplied, so no cost is shown.</p>';
    return `<section class="report-section">${sectionHead(hl, 'Threshold outcomes', 'What each model does at the threshold chosen on validation data, applied unchanged to the test partition.')}${table(ids, 'Outcomes on the test partition at each model threshold', headers, rows)}${costNote}<p class="notice">Thresholds were chosen on the validation partition only. Changing one after looking at these results is exploration, not a validated choice.</p>${table(ids, 'How each threshold was selected', [{ text: 'Model' }, { text: 'Rule' }, { text: 'Validation rows', num: true }, { text: 'Test labels used' }], sel)}${budgetRows.length ? table(ids, 'Review budgets: the top-scored cases on the test partition', [{ text: 'Model' }, { text: 'Policy' }, { text: 'Share of test', num: true }, { text: 'Flagged', num: true }, { text: 'Found', num: true }, { text: 'False alarms', num: true }, { text: 'Precision', num: true }, { text: 'Recall', num: true }], budgetRows) : ''}${ties.length ? `<ul class="tie-notes">${ties.map(t => `<li>${esc(t)}</li>`).join('')}</ul>` : ''}${term('threshold')}${term('precision_recall')}${term('review_budget')}${costed ? term('cost') : ''}</section>`;
  }

  function renderProbability(vm, ids, hl) {
    const ok = vm.models.filter(m => typeof m.metrics.brier_score === 'number' || typeof m.metrics.log_loss === 'number');
    if (!ok.length) return `<section class="report-section">${sectionHead(hl, 'Probability diagnostics')}<p>Probability diagnostics are not available for these results.</p></section>`;
    const cal = vm.models.map(m => (m.execution && m.execution.calibration) || 'none');
    const rows = ok.map(m => [m.name, U.fixed(m.metrics.brier_score), U.fixed(m.metrics.log_loss), (m.execution && m.execution.calibration) || 'none']);
    const bins = ok.filter(m => m.calibration && Array.isArray(m.calibration.bins)).map(m => `<details class="more"><summary>Reliability bins: ${esc(m.name)}</summary>${table(ids, `Reliability bins for ${m.name}`, [{ text: 'Score range' }, { text: 'Cases', num: true }, { text: 'Mean score', num: true }, { text: 'Observed positive share', num: true }], m.calibration.bins.map(b => [`${U.fixed(b.lower, 1)} to ${U.fixed(b.upper, 1)}${b.upper_inclusive ? ' (inclusive)' : ''}`, U.int(b.count), U.fixed(b.mean_score), U.fixed(b.observed_positive_fraction)]))}<p class="table-note">${esc(m.calibration.interpretation || 'Descriptive reliability bins; calibration is not certified.')}</p></details>`).join('');
    return `<section class="report-section">${sectionHead(hl, 'Probability diagnostics', 'Shown because these scores are probabilities for the positive class. Lower Brier score and log loss are better.')}${table(ids, 'Brier score and log loss on the test partition', [{ text: 'Model' }, { text: 'Brier score', num: true }, { text: 'Log loss', num: true }, { text: 'Calibration step' }], rows)}<p class="table-note">These do not certify calibration. ${cal.includes('sigmoid') ? 'Sigmoid calibration was fitted inside the train partition.' : 'No calibration step was applied.'}</p>${bins}${term('brier')}${term('reliability')}</section>`;
  }

  function renderCost(vm, ids, hl) {
    if (!vm.models.length) return '';
    const rows = vm.models.map(m => [m.name, U.seconds(m.execution.fit_seconds), U.seconds(m.wall), U.megabytes(m.execution.peak_process_rss_bytes), m.cost, m.execution.threads == null ? U.UNAVAILABLE : m.execution.threads]);
    return `<section class="report-section">${sectionHead(hl, 'Time and resource cost', 'Measured on the machine that ran this analysis, with one worker thread per model.')}${table(ids, 'Time and memory per model', [{ text: 'Model' }, { text: 'Fit time', num: true }, { text: 'Worker wall time', num: true }, { text: 'Peak process memory', num: true }, { text: 'Cost class' }, { text: 'Threads', num: true }], rows)}<p class="table-note">One run on a shared machine is not a benchmark. Process memory includes the interpreter and libraries. The cost class is a qualitative registry label.</p></section>`;
  }

  function renderLimits(vm, hl) {
    const own = ['One split, no uncertainty interval, and no automatic winner: close models can swap order under another split.',
      'The app cannot check whether your columns would be known at prediction time, or the quality of your labels.',
      'Probability metrics are descriptive and do not certify calibration.'];
    if (vm.failures.length) own.push(`Models without a result (${vm.failures.map(f => f.name).join(', ')}) are part of the comparison and are listed above.`);
    const fromReport = vm.first ? ((vm.result.outcomes[vm.first.id].report || {}).limitations || []) : [];
    const warn = [...new Set(vm.models.flatMap(m => m.warnings || []))];
    return `<section class="report-section">${sectionHead(hl, 'Limits of this report')}<ul>${own.map(t => `<li>${esc(t)}</li>`).join('')}</ul>${warn.length ? `<h${hl + 1}>Warnings from the run</h${hl + 1}><ul>${warn.map(t => `<li>${esc(t)}</li>`).join('')}</ul>` : ''}${fromReport.length ? `<details class="more"><summary>Limitations recorded in the report data</summary><ul>${fromReport.map(t => `<li>${esc(t)}</li>`).join('')}</ul></details>` : ''}</section>`;
  }

  function renderTrainingReport(result, options = {}) {
    const vm = trainingViewModel(result);
    const hl = options.headingLevel || 3, key = provenanceKey(options.provenance || result.provenance, 'fresh_app_run');
    const ids = makeIds(options.idPrefix || 'r'); ids.prefix = options.idPrefix || 'r';
    const meta = [vm.tool && `Package ${vm.tool.name || 'rare-event-eval'} ${vm.tool.version || ''}`.trim(), result.schema_version && `schema ${result.schema_version}`, vm.created && `created ${vm.created}`].filter(Boolean).map(esc).join(' | ');
    return `<div class="report" data-provenance="${esc(provenanceKey(key))}">${badge(key, options.provenanceNote)}${meta ? `<p class="report-meta">${meta}</p>` : ''}<p class="aggregate-note">Aggregate report: no rows, predictions or per-case scores are included.</p>${renderDatasetSection(vm, ids, hl)}${renderModelsSection(vm, ids, hl)}${renderRanking(vm, ids, hl)}${renderThresholds(vm, ids, hl)}${renderProbability(vm, ids, hl)}${renderCost(vm, ids, hl)}${renderLimits(vm, hl)}</div>`;
  }

  /* ---- saved-prediction report (schema 1.0) ---- */
  function statusText(value) {
    return ({ not_checked: 'Not checked', not_estimated: 'Not estimated', row_indices_disjoint: 'Training and test rows are disjoint' }[value]
      || (String(value).startsWith('not_certified:') ? 'Data availability and domain semantics still need review' : value));
  }
  function renderPredictionReport(report, options = {}) {
    const view = options.view || 'budget', hl = options.headingLevel || 3, key = provenanceKey(options.provenance, 'fresh_app_run');
    const ids = makeIds(options.idPrefix || 'p'); ids.prefix = options.idPrefix || 'p';
    const data = report.data, entries = Object.entries(report.models);
    const withAp = entries.filter(([, m]) => typeof m.metrics.average_precision === 'number');
    const apChart = bars(ids, 'Average precision on these cases', entries.map(([name, m]) => ({ label: name, value: m.metrics.average_precision })), {
      baseline: { value: data.positive_prevalence || 0, label: `no-skill baseline (positive share ${U.pct(data.positive_prevalence)})` },
      summary: withAp.length ? `Average precision ranges from ${U.fixed(Math.min(...withAp.map(([, m]) => m.metrics.average_precision)))} to ${U.fixed(Math.max(...withAp.map(([, m]) => m.metrics.average_precision)))}.` : 'No average precision is available.' });
    const rows = entries.map(([name, m]) => {
      const d = view === 'threshold' ? m.threshold_result : m.review_budget_results[0];
      if (!d) return [name, 'No decision policy supplied.', '', '', '', '', '', ''];
      return [name, view === 'threshold' ? `Score >= ${U.num(d.threshold)}` : `Top ${U.int(d.budget)}`, U.int(d.flagged), U.int(d.true_positives), U.int(d.false_positives), U.int(d.false_negatives), U.pct(d.precision), U.pct(d.recall)];
    });
    const metricRows = entries.map(([name, m]) => [name, U.fixed(m.metrics.average_precision), U.fixed(m.metrics.roc_auc), U.fixed(m.metrics.brier_score), U.fixed(m.metrics.log_loss), m.threshold_result && typeof m.threshold_result.assumed_total_cost === 'number' ? U.num(m.threshold_result.assumed_total_cost, 2) : U.UNAVAILABLE]);
    const notes = [];
    entries.forEach(([name, m]) => {
      (m.warnings || []).forEach(n => notes.push(`${name}: ${n}`));
      m.review_budget_results.forEach(r => { if (r.boundary_tie) notes.push(`${name}, top ${r.budget}: tied scores can yield ${r.boundary_tie.true_positives_min} to ${r.boundary_tie.true_positives_max} positives found, depending on which tied cases are selected. These are allocation bounds, not confidence intervals.`); });
    });
    const provenance = report.task.evaluation_role === 'test' && report.execution_mode === 'fresh_training'
      ? 'Fresh fit with separate training, validation and test partitions. Thresholds were chosen using validation labels only.'
      : 'Saved predictions. Class orientation and how the predictions were produced are supplied by you and are not verified by this app.';
    const costed = entries.some(([, m]) => m.threshold_result && typeof m.threshold_result.assumed_total_cost === 'number');
    const tiles = `<div class="stats">${stat('Evaluated cases', U.int(data.rows))}${stat('Positive cases', U.int(data.positive_rows))}${stat('Positive prevalence', U.pct(data.positive_prevalence))}</div>`;
    return `<div class="report" data-provenance="${esc(key)}">${badge(key, options.provenanceNote)}<p class="aggregate-note">Aggregate report: no rows or per-case scores are included.</p>${tiles}
      <section class="report-section">${sectionHead(hl, 'Ranking quality', 'Average precision on the same cases for every score column. This is not a significance test or an automatic model recommendation.')}${apChart}${term('ap')}</section>
      <section class="report-section">${sectionHead(hl, view === 'threshold' ? 'Threshold decisions' : 'What fits the review budget?')}${table(ids, view === 'threshold' ? 'Outcomes at the fixed threshold' : 'Outcomes at the review budget', [{ text: 'Model' }, { text: 'Policy' }, { text: 'Flagged', num: true }, { text: 'Found', num: true }, { text: 'False alarms', num: true }, { text: 'Missed', num: true }, { text: 'Precision', num: true }, { text: 'Recall', num: true }], rows)}<p class="table-note">Positive: <strong>${esc(report.task.positive_label)}</strong>. Negative: <strong>${esc(report.task.negative_label)}</strong>. Top-k score ties use original row order.</p>${term('review_budget')}${term('threshold')}</section>
      <section class="report-section">${sectionHead(hl, 'Additional metrics and cost assumptions')}${table(ids, 'Additional metrics', [{ text: 'Model' }, { text: 'AP', num: true }, { text: 'ROC AUC', num: true }, { text: 'Brier', num: true }, { text: 'Log loss', num: true }, { text: 'Threshold cost', num: true }], metricRows)}<p class="table-note">${costed ? 'Cost is computed from the assumptions you supplied. These are assumptions, not measured savings. ' : 'No cost assumptions were supplied. '}Brier score and log loss are unavailable for ranking scores. Reliability bins are in the JSON.</p>${term('brier')}${costed ? term('cost') : ''}</section>
      ${notes.length ? `<section class="report-section callout callout-warn">${sectionHead(hl, 'Read before deciding')}<ul>${[...new Set(notes)].map(n => `<li>${esc(n)}</li>`).join('')}</ul></section>` : ''}
      <section class="report-section">${sectionHead(hl, 'What this report establishes')}<p>${esc(provenance)}</p><dl class="facts"><div><dt>Label and score checks</dt><dd>Performed</dd></div><div><dt>Training leakage</dt><dd>${esc(statusText(report.checks.training_leakage))}</dd></div><div><dt>Row and split overlap</dt><dd>${esc(statusText(report.checks.train_test_overlap))}</dd></div><div><dt>Uncertainty</dt><dd>${esc(statusText(report.checks.uncertainty))}</dd></div></dl><p class="table-note">${esc(report.data.description || '')} ${esc(report.policies.selection || '')}</p>${options.cannotCertify ? `<p class="notice">${esc(options.cannotCertify)}</p>` : ''}</section>
      ${Array.isArray(report.limitations) ? `<details class="more"><summary>Limitations recorded in the report</summary><ul>${report.limitations.map(n => `<li>${esc(n)}</li>`).join('')}</ul></details>` : ''}</div>`;
  }

  /* A standalone HTML document for a saved-prediction report: no script, no network asset, escaped values. */
  const DOCUMENT_STYLE = 'body{font-family:system-ui,sans-serif;max-width:1050px;margin:32px auto;padding:20px;color:#1d302d;background:#fff;line-height:1.5}h1,h2,h3{line-height:1.25}table{width:100%;border-collapse:collapse}caption{text-align:left;font-weight:600;padding:6px 0}td,th{padding:8px;text-align:left;border-bottom:1px solid #b7c2bb;vertical-align:top}.num{text-align:right}.stats{display:flex;gap:20px;flex-wrap:wrap}.stat{padding:10px;border:1px solid #b7c2bb}.stat-label,.stat-sub{display:block}.stat-value{font-size:1.6rem}.bars{list-style:none;padding:0}.bar-row{display:grid;grid-template-columns:160px 1fr 60px;gap:12px;margin:8px 0;align-items:center}.bar{width:100%;height:12px}.bar-track{fill:#e6ebe7}.bar-fill{fill:#1f5a49}.bar-base{fill:#8a3d00}.table-scroll{overflow:auto}.badge{border:1px solid #1f5a49;padding:3px 8px}.visually-hidden{position:absolute;left:-9999px}.notice,.callout{border-left:4px solid #8a3d00;padding:8px 12px;background:#f6efe4}.table-note,.muted{color:#4f5f57}details{margin:8px 0}';
  function predictionReportDocument(report, options = {}) {
    const key = provenanceKey(options.provenance, 'fresh_app_run');
    const body = renderPredictionReport(report, { ...options, provenance: key, headingLevel: 2, idPrefix: 'doc' });
    const toolName = report.tool && report.tool.name ? report.tool.name : 'rare-event-eval-browser';
    return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Rare-event evaluation report</title><style>${DOCUMENT_STYLE}</style></head><body><main><h1>Rare-event evaluation report</h1><p>Sherwan Ali | ${esc(toolName)} ${esc(report.tool && report.tool.version)}</p>${body}</main></body></html>`;
  }

  /* A small card for the landing page: provenance, three numbers and the ranking bars. */
  function renderPreview(source, options = {}) {
    const ids = makeIds('pv');
    if (source && source.outcomes) {
      const vm = trainingViewModel(source), key = provenanceKey(options.provenance || source.provenance, 'recorded_app_run');
      const prevalence = vm.first ? vm.first.test.positive_prevalence : null;
      return `<div class="preview-card">${badge(key)}<div class="stats">${stat('Rows used', U.int(vm.counts.effective_rows))}${stat('Models', U.int(vm.models.length + vm.failures.length))}${stat('Test positives', U.int(vm.first && vm.first.test.positive_rows))}</div>${bars(ids, 'Average precision on the test partition', vm.models.map(m => ({ label: m.name, value: m.metrics.average_precision })), { baseline: { value: typeof prevalence === 'number' ? prevalence : 0, label: 'no-skill baseline' } })}<p class="comparison-statement">${esc(comparisonStatement(vm.models).text)}</p></div>`;
    }
    const entries = Object.entries(source.models), key = provenanceKey(options.provenance, 'fictional_fixture');
    return `<div class="preview-card">${badge(key)}<div class="stats">${stat('Cases', U.int(source.data.rows))}${stat('Positive', U.int(source.data.positive_rows))}${stat('Score columns', U.int(entries.length))}</div>${bars(ids, 'Average precision on these cases', entries.map(([n, m]) => ({ label: n, value: m.metrics.average_precision })), { baseline: { value: source.data.positive_prevalence || 0, label: 'no-skill baseline' } })}</div>`;
  }

  const api = { trainingViewModel, comparisonStatement, renderTrainingReport, renderPredictionReport, predictionReportDocument, renderPreview,
    provenanceKey, badge, term, table, bars, stat, capacityText, makeIds, TIE_AP, FEW_POSITIVES };
  root.FYD.reportView = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
