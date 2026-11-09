/* Sherwan Ali. Pure page renderers (state in, HTML text out): Overview, the five steps of "Analyze a dataset",
   Research and How it works. Every value from a file, a server or a report is escaped. No DOM, no network. */
(function (root) {
  'use strict';
  const U = root.FYD.util, C = root.FYD.content, RV = root.FYD.reportView, MODELS = root.FYD.MODELS, FLOW = root.FYD.flow, esc = U.esc;
  const STEP_TITLES = { data: 'Data', target: 'Target and split', models: 'Models and policy', run: 'Run', results: 'Results and export' };
  const STEP_ORDER = FLOW.STEPS;

  /* ---- small form helpers ---- */
  const errFor = (state, id) => state.errors.find(e => e.field === id) || null;
  function describe(state, id, hasHint) {
    const parts = [];
    if (hasHint) parts.push(`${id}-hint`);
    const e = errFor(state, id);
    if (e) parts.push(`${id}-error`);
    return `${parts.length ? ` aria-describedby="${esc(parts.join(' '))}"` : ''}${e ? ' aria-invalid="true"' : ''}`;
  }
  function field(state, id, label, control, hint) {
    const e = errFor(state, id);
    return `<div class="field${e ? ' has-error' : ''}"><label for="${esc(id)}">${esc(label)}</label>${hint ? `<p class="hint" id="${esc(id)}-hint">${esc(hint)}</p>` : ''}${e ? `<p class="field-error" id="${esc(id)}-error"><span class="visually-hidden">Error: </span>${esc(e.message)}</p>` : ''}${control}</div>`;
  }
  function select(state, id, options, hasHint, extra) {
    return `<select id="${esc(id)}"${extra || ''}${describe(state, id, hasHint)}>${options.map(o => `<option value="${esc(o.value)}"${o.selected ? ' selected' : ''}>${esc(o.label)}</option>`).join('')}</select>`;
  }
  function input(state, id, type, value, hasHint, extra) {
    return `<input id="${esc(id)}" type="${esc(type)}" value="${esc(value)}"${extra || ''}${describe(state, id, hasHint)}>`;
  }
  function errorSummary(state) {
    if (!state.errors.length) return '';
    return `<div class="error-summary" id="error-summary" role="alert" tabindex="-1"><h3>There ${state.errors.length === 1 ? 'is a problem' : 'are problems'}</h3><ul>${state.errors.map(e => `<li><a href="#${esc(e.field)}" data-action="focus-field" data-field="${esc(e.field)}">${esc(e.message)}</a></li>`).join('')}</ul></div>`;
  }
  const button = (action, label, cls, extra) => `<button type="button" class="${esc(cls || 'btn')}" data-action="${esc(action)}"${extra || ''}>${esc(label)}</button>`;
  const modelById = id => MODELS.find(m => m.id === id) || null;
  const modelName = id => (modelById(id) || { name: id }).name;

  /* ---- stepper ---- */
  function renderStepper(state, flow) {
    const items = STEP_ORDER.map((step, i) => {
      const current = state.step === step, done = flow.stepComplete(step) && STEP_ORDER.indexOf(state.step) > i;
      const reachable = current || (i < STEP_ORDER.indexOf(state.step)) || flow.canReach(step);
      const mark = done ? 'Done' : current ? 'Current' : (reachable ? 'Available' : 'Locked');
      return `<li class="step${current ? ' is-current' : ''}${done ? ' is-done' : ''}"><button type="button" class="step-button" data-action="goto" data-step="${esc(step)}" aria-label="${esc(`Step ${i + 1} of ${STEP_ORDER.length}: ${STEP_TITLES[step]}, ${mark.toLowerCase()}`)}"${current ? ' aria-current="step"' : ''}${reachable ? '' : ' disabled'}><span class="step-number" aria-hidden="true">${i + 1}</span> <span class="step-title">${esc(STEP_TITLES[step])}</span> <span class="step-state">${esc(mark)}</span></button></li>`;
    }).join('');
    return `<nav aria-label="Analysis steps"><ol class="steps">${items}</ol></nav>`;
  }

  /* ---- step 1: data ---- */
  function limitsTable(limits) {
    const rows = [['Upload size', U.megabytes(limits.upload_limit_bytes)], ['Rows', U.int(limits.max_rows)], ['Expanded predictor columns', U.int(limits.max_features)], ['Models per run', U.int(limits.max_models_per_job)],
      ['Time per model', U.duration(limits.per_model_timeout_seconds)], ['Time per run', U.duration(limits.job_timeout_seconds)], ['Wait in the queue', U.duration(limits.queue_wait_seconds)],
      ['Upload kept without use', U.duration(limits.dataset_idle_seconds)], ['Finished run kept', U.duration(limits.finished_job_seconds)]];
    return RV.table(RV.makeIds('lim'), 'Service limits, provisional presets', [{ text: 'Limit' }, { text: 'Value', num: true }], rows);
  }
  function renderProfileSummary(state) {
    const p = state.data.profile, d = p.parsing || {}, lim = (p.limits || {});
    const notes = [].concat(d.format_notes || [], d.read_warnings || []);
    const skipped = d.malformed_rows_skipped_by_reader;
    const status = lim.status === 'ok' ? '<p><span class="status status-ok">Within the service limits</span></p>'
      : `<div class="callout callout-warn" role="group" aria-label="Limit status"><p><span class="status status-bad">${esc(lim.status === 'exceeded' ? 'Over a service limit' : 'Not valid as it stands')}</span> ${esc(lim.code || '')}${lim.code && C.REASONS[lim.code] ? ': ' + esc(C.REASONS[lim.code]) : ''}</p>${lim.message ? `<p>${esc(lim.message)}</p>` : ''}${(lim.violations || []).length ? `<ul>${lim.violations.map(v => `<li>${esc(typeof v === 'string' ? v : JSON.stringify(v))}</li>`).join('')}</ul>` : ''}${lim.status_if_suggested_exclusions_applied ? `<p>With the suggested exclusions applied the status would be <strong>${esc(lim.status_if_suggested_exclusions_applied.status)}</strong>.</p>` : ''}</div>`;
    return `<section class="card" aria-labelledby="profile-heading"><h3 id="profile-heading">What the service read</h3><div class="stats">${RV.stat('Rows', U.int(p.rows))}${RV.stat('Columns', U.int(p.columns))}${RV.stat('Input size', U.size(d.input_bytes))}</div>${status}
      ${notes.length ? `<h4>Parsing notes</h4><ul>${notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>` : '<p>The reader reported no format notes or warnings.</p>'}${typeof skipped === 'number' && skipped > 0 ? `<p class="notice">The reader skipped ${U.int(skipped)} malformed rows and counted them in the warnings above.</p>` : ''}
      ${d.note ? `<p class="muted">${esc(d.note)}</p>` : ''}<details class="more"><summary>Service limits</summary>${state.data.limits ? limitsTable(state.data.limits) : ''}</details></section>`;
  }
  function renderPrivacyHosted(state) {
    const l = state.data.limits;
    const text = l ? C.PRIVACY.hostedAfter({ datasetIdle: U.duration(l.dataset_idle_seconds), finishedJob: U.duration(l.finished_job_seconds) }) : C.PRIVACY.hostedBefore;
    return `<aside class="privacy" aria-label="Privacy in this mode"><h3>Privacy: hosted analysis</h3><p>${esc(text)}</p><p>${esc(C.PRIVACY.downloads)}</p></aside>`;
  }
  function renderData(state) {
    const d = state.data, s = state.samples, src = d.source;
    const samples = s.phase === 'ready' && s.items.length ? `<fieldset class="sample-list"><legend>Or use a bundled sample</legend>${s.items.map(it => `<div class="radio-row"><input type="radio" name="sample" id="sample-${esc(it.name)}" value="${esc(it.name)}" data-action="pick-sample"${src && src.kind === 'sample' && src.name === it.name ? ' checked' : ''}><label for="sample-${esc(it.name)}"><strong>${esc(it.label)}</strong> <span class="pill">${esc(it.category)}</span><span class="radio-desc">${esc(it.desc)}</span></label></div>`).join('')}</fieldset>`
      : s.phase === 'loading' ? '<p class="muted">Loading the bundled samples.</p>' : s.phase === 'error' ? `<div class="notice" role="note"><p>The sample list could not be loaded. ${esc(s.error)}</p><p><button type="button" class="btn" id="retry-samples" data-action="retry-samples">Try again</button></p></div>` : '';
    const selected = src ? `<p class="selected-source" id="selected-source"><strong>${d.selectedNotUploaded ? 'Selected, not uploaded yet:' : 'Uploaded:'}</strong> ${esc(src.kind === 'sample' ? src.label : src.name)}${src.size != null ? ` (${esc(U.size(src.size))})` : ''}</p>` : '<p class="muted" id="selected-source">Nothing selected yet.</p>';
    const expired = d.phase === 'expired' ? `<p class="notice" role="note">${esc(d.error)}</p>` : '';
    const busy = d.phase === 'uploading';
    return `<section aria-labelledby="step-heading"><h2 id="step-heading" tabindex="-1">1. Data</h2><p class="section-intro">Pick a file or a bundled sample. Nothing is sent until you choose Upload and profile.</p>${expired}
      <div class="dropzone" id="dropzone"><label for="file-input"><span class="dropzone-title">Choose a data file</span></label><p class="hint" id="file-input-hint">or drop it on this area. CSV, TSV, Excel, JSON, ARFF, Stata, ZIP or GZIP. The service sets the size limit; it is shown after the upload.</p><input type="file" id="file-input" accept=".csv,.tsv,.txt,.data,.dat,.names,.xlsx,.xls,.json,.dta,.arff,.zip,.gz,.tab" aria-describedby="file-input-hint${errFor(state, 'file-input') ? ' file-input-error' : ''}">${errFor(state, 'file-input') ? `<p class="field-error" id="file-input-error"><span class="visually-hidden">Error: </span>${esc(errFor(state, 'file-input').message)}</p>` : ''}</div>
      ${samples}${selected}
      ${d.importChoices ? `<div class="field"><label for="import-table">Choose the ${d.importChoices.kind==='sheet'?'worksheet':'archive member'} to analyze</label><select id="import-table" data-action="import-selection" data-kind="${esc(d.importChoices.kind)}"><option value="">Select a table, then upload and profile again</option>${d.importChoices.choices.map(n=>`<option value="${esc(n)}"${(d.importSelection||{})[d.importChoices.kind]===n?' selected':''}>${esc(n)}</option>`).join('')}</select></div>` : ''}
      <div class="actions">${button('upload', busy ? 'Uploading' : 'Upload and profile', 'btn btn-primary', busy ? ' disabled' : '')}${d.phase === 'ready' ? button('delete-ask', 'Delete my uploaded data', 'btn btn-quiet') : ''}</div>
      ${state.ui.confirmDelete ? renderDeleteConfirm() : ''}
      ${d.phase === 'ready' ? renderProfileSummary(state) : ''}
      ${renderPrivacyHosted(state)}
      <div class="actions step-actions">${d.phase === 'ready' ? `<button type="button" class="btn btn-primary" id="next-data" data-action="next">Continue to target and split</button>` : ''}</div></section>`;
  }
  function renderDeleteConfirm() {
    return `<div class="callout callout-warn" role="group" aria-labelledby="del-title"><h3 id="del-title">Delete your uploaded data?</h3><p>This removes the upload and every run on it from the service, and clears the results from this page. Download what you need first. This cannot be undone.</p><div class="actions">${button('delete-confirm', 'Delete now', 'btn btn-danger')}${button('delete-cancel', 'Keep it', 'btn')}</div></div>`;
  }

  /* ---- step 2: target and split ---- */
  function renderClassBlock(state) {
    const t = state.target, cls = state.classes;
    if (!t.column) return '';
    const lab = cls && cls.labels;
    let counts = '';
    if (lab) {
      const total = lab.reduce((a, l) => a + l.count, 0);
      counts = RV.table(RV.makeIds('cls'), `Labels in ${t.column}${cls.truncated ? ' (first 50)' : ''}`, [{ text: 'Label' }, { text: 'Rows', num: true }, { text: 'Share', num: true }], lab.map(l => [l.label, U.int(l.count), U.pct(total ? l.count / total : null)]))
        + (cls.missing ? `<p class="notice">${U.int(cls.missing)} rows have no value in the target column.</p>` : '');
    } else {
      counts = '<p class="notice">Label counts for this column are not available before the run. Enter the labels exactly as they appear in the file.</p>';
    }
    const suggestedPos = state.data.profile && state.data.profile.classes && state.data.profile.classes.column === t.column && state.data.profile.classes.suggested_positive;
    let posControl, negControl = '';
    if (lab) {
      posControl = select(state, 'positive-label', [{ value: '', label: 'Choose the positive class', selected: !t.positive }, ...lab.map(l => ({ value: l.label, label: `${l.label} (${U.int(l.count)} rows)`, selected: l.label === t.positive }))], true);
      if (cls.distinct === 2) negControl = `<p class="derived">Negative class: <strong>${esc(t.negative || 'not set')}</strong> (the other label)</p>`;
      else negControl = `<p class="derived">${U.int(cls.distinct)} labels: every label other than ${esc(t.positive || 'the positive class')} is grouped as the negative class (one against all others). The report lists each label that was merged.</p>`;
    } else {
      posControl = input(state, 'positive-label', 'text', t.positive, true);
      negControl = `${field(state, 'negative-label', 'Negative class label', input(state, 'negative-label', 'text', t.negative, false, t.oneVsRest ? ' disabled' : ''))}<div class="check-row"><input type="checkbox" id="one-vs-rest"${t.oneVsRest ? ' checked' : ''} data-action="set-ovr"><label for="one-vs-rest">The column has more than two labels: use the positive class against all others</label></div>`;
    }
    const posHint = suggestedPos ? `Suggested, please confirm: ${suggestedPos.label}. ${suggestedPos.basis}.` : 'The rarest label is not assumed for you. Choose the label of the cases you want to find.';
    return `<h3>Classes</h3>${counts}${field(state, 'positive-label', 'Positive class', posControl, posHint)}${negControl}`;
  }
  function renderColumns(state) {
    const p = state.data.profile, t = state.target;
    const cols = p.schema_columns || [];
    const sugg = new Map((p.suggested_exclusions || []).map(x => [x.column, x.reason]));
    const rows = cols.map(c => {
      const fixed = c.column === t.column ? 'Target' : c.column === t.groupColumn ? 'Group column' : c.column === t.timeColumn ? 'Time column' : null;
      const role = t.roles[c.column] || 'include';
      const control = fixed ? `<span class="pill">${esc(fixed)}</span>` : `<label class="visually-hidden" for="role-${esc(c.column)}">Role of column ${esc(c.column)}</label><select id="role-${esc(c.column)}" data-action="set-role" data-column="${esc(c.column)}"><option value="include"${role === 'include' ? ' selected' : ''}>Use as predictor</option><option value="exclude"${role === 'exclude' ? ' selected' : ''}>Leave out</option><option value="id"${role === 'id' ? ' selected' : ''}>Identifier, never a predictor</option></select>`;
      return [c.column, c.dtype, U.int(c.unique), U.int(c.missing), { html: control }, sugg.has(c.column) ? sugg.get(c.column) : ''];
    });
    const explain = sugg.size ? `<div class="callout" role="note"><p><strong>Suggested exclusions, pre-selected.</strong> ${U.int(sugg.size)} ${sugg.size === 1 ? 'column looks' : 'columns look'} like identifiers or free text because more than half of the rows hold distinct values. The older app dropped such columns without asking. Here they are pre-selected as Leave out, explained in the last column, and you can change each one.</p></div>` : '<p class="muted">No column was flagged as identifier-like.</p>';
    return `<h3>Predictors</h3><p class="section-intro">Every column except the target, group and time columns is a predictor unless you leave it out. Leave out columns that would not be known when the model is used, or that are identifiers.</p>${explain}${RV.table(RV.makeIds('col'), 'Columns and their role', [{ text: 'Column' }, { text: 'Type' }, { text: 'Distinct values', num: true }, { text: 'Missing', num: true }, { text: 'Role' }, { text: 'Note' }], rows)}`;
  }
  function renderDiagnostics(state) {
    const pp = state.data.profile && state.data.profile.package_profile;
    if (!pp) return '';
    const dup = pp.duplicate_feature_groups || {}, exp = pp.expanded_feature_estimate || {};
    const list = (label, items) => `<div><dt>${esc(label)}</dt><dd>${items && items.length ? U.listText(items) : 'none'}</dd></div>`;
    return `<details class="more"><summary>Data diagnostics</summary><dl class="facts">${list('Constant columns', pp.constant_columns)}${list('Empty columns', pp.empty_columns)}${list('Duplicate columns', (pp.duplicate_columns || []).map(x => Array.isArray(x) ? x.join(' = ') : String(x)))}<div><dt>Repeated feature patterns</dt><dd>${U.int(dup.groups)} groups covering ${U.int(dup.rows_in_groups)} rows. ${esc(dup.interpretation || '')}</dd></div><div><dt>Expanded predictor columns (upper bound)</dt><dd>${U.int(exp.total)}</dd></div></dl></details>`;
  }
  function renderTarget(state) {
    const t = state.target, p = state.data.profile, cols = (p.schema_columns || []).map(c => c.column);
    const suggestion = p.target && p.target.suggestion && p.target.suggestion.column;
    const suggestedNote = suggestion && suggestion === t.column ? `Suggested by the profile, please confirm.${p.target.suggestion.confident === false ? ' The profile is not confident about this suggestion.' : ''}` : (t.column ? 'Chosen by you.' : 'No column chosen.');
    const colOptions = [{ value: '', label: 'Choose the target column', selected: !t.column }, ...cols.map(c => ({ value: c, label: c, selected: c === t.column }))];
    const optional = (id, label, current, hint) => field(state, id, label, select(state, id, [{ value: '', label: 'None', selected: !current }, ...cols.filter(c => c !== t.column).map(c => ({ value: c, label: c, selected: c === current }))], true), hint);
    const splits = `<fieldset class="split-list"><legend>What should the model generalize to?</legend><p class="hint">This decides how rows are separated into train, validation and test.</p>${C.SPLIT_MODES.map(m => `<div class="radio-row"><input type="radio" name="split-mode" id="split-${esc(m.id)}" value="${esc(m.id)}" data-action="set-split"${t.splitMode === m.id ? ' checked' : ''}${errFor(state, 'split-mode') && m.id === C.SPLIT_MODES[0].id ? ' aria-invalid="true"' : ''}><label for="split-${esc(m.id)}"><strong>${esc(m.name)}</strong><span class="radio-desc">${esc(m.question)}</span><span class="radio-desc muted">${esc(m.detail)}</span></label></div>`).join('')}</fieldset>`;
    const needs = t.splitMode === 'group' ? optional('group-column', 'Group column (required for this split)', t.groupColumn, 'Rows that share a value stay together in one partition.')
      : t.splitMode === 'chronological' ? optional('time-column', 'Time column (required for this split)', t.timeColumn, 'Numeric values or dates. Rows are ordered by this column.') : '';
    const miss = `<div class="field"><label for="missing-policy">Rows with no target value</label><select id="missing-policy" data-action="set-missing"><option value="error"${t.missingPolicy === 'error' ? ' selected' : ''}>Stop with an error</option><option value="drop_reported"${t.missingPolicy === 'drop_reported' ? ' selected' : ''}>Exclude them and list them in the report</option></select></div>`;
    return `<section aria-labelledby="step-heading"><h2 id="step-heading" tabindex="-1">2. Target and split</h2><p class="section-intro">Confirm what is being predicted. Nothing here starts a run.</p>
      <div class="card"><h3>Target</h3>${field(state, 'target-column', 'Target column', select(state, 'target-column', colOptions, true, ' data-action="set-target"'), suggestedNote)}${renderClassBlock(state)}</div>
      <div class="card">${renderColumns(state)}${renderDiagnostics(state)}</div>
      <div class="card"><h3>Split</h3>${splits}${needs}
        ${field(state, 'seed', 'Seed', input(state, 'seed', 'text', t.seed, true, ' inputmode="numeric"'), 'A whole number. The same data, settings and seed give the same split.')}${miss}</div>
      <div class="card"><div class="check-row"><input type="checkbox" id="confirm-target"${t.confirmed ? ' checked' : ''} data-action="confirm-target"${describe(state, 'confirm-target', false)}><label for="confirm-target">I confirm the target column, the positive class and the columns I leave out.</label></div>${errFor(state, 'confirm-target') ? `<p class="field-error" id="confirm-target-error">${esc(errFor(state, 'confirm-target').message)}</p>` : ''}</div>
      <div class="actions step-actions">${button('back', 'Back', 'btn')}${button('next', 'Continue to models and policy', 'btn btn-primary')}</div></section>`;
  }

  /* ---- step 3: models and policy ---- */
  function renderModels(state) {
    const m = state.models, lim = state.data.limits || {};
    const list = MODELS.map(mod => `<div class="check-row model-row"><input type="checkbox" id="model-${esc(mod.id)}" value="${esc(mod.id)}" data-action="toggle-model"${m.selected.includes(mod.id) ? ' checked' : ''} aria-describedby="model-${esc(mod.id)}-desc"><label for="model-${esc(mod.id)}"><strong>${esc(mod.name)}</strong> <span class="pill">${esc(mod.cost)}</span>${mod.default ? ' <span class="pill pill-quiet">package default set</span>' : ''}</label><p class="hint" id="model-${esc(mod.id)}-desc">${esc(mod.description)} <span class="mono">${esc(mod.recipe)}</span></p></div>`).join('');
    const calib = field(state, 'calibration', 'Probability calibration', select(state, 'calibration', C.CALIBRATIONS.map(c => ({ value: c.id, label: c.name, selected: m.calibration === c.id })), true, ' data-action="set-calibration"'), (C.CALIBRATIONS.find(c => c.id === m.calibration) || {}).detail);
    const budgets = field(state, 'review-budgets', 'Review budgets, percent of test cases', input(state, 'review-budgets', 'text', m.reviewPercents, true, ''), 'Separate values with commas, for example 5, 10, 20. Each budget flags that share of the test cases with the highest scores. Leave blank for none.');
    const costs = `<fieldset><legend>Optional cost assumptions</legend><p class="hint">Use one unit throughout. These are assumptions, not measured savings. When given, each model threshold on the validation partition minimizes the assumed cost instead of maximizing F1.</p>${field(state, 'analyze-cost-fp', 'Cost per false alarm', input(state, 'analyze-cost-fp', 'text', m.costs.fp, false, ' inputmode="decimal" data-action="set-cost" data-key="fp"'))}${field(state, 'analyze-cost-fn', 'Cost per missed positive', input(state, 'analyze-cost-fn', 'text', m.costs.fn, false, ' inputmode="decimal" data-action="set-cost" data-key="fn"'))}${field(state, 'analyze-cost-review', 'Cost per flagged case, optional', input(state, 'analyze-cost-review', 'text', m.costs.review, false, ' inputmode="decimal" data-action="set-cost" data-key="review"'))}</fieldset>`;
    const limitText = lim.per_model_timeout_seconds ? `Each model has up to ${U.duration(lim.per_model_timeout_seconds)} and the whole run up to ${U.duration(lim.job_timeout_seconds)}. A model that reaches its limit is reported as timed out, not dropped. Up to ${U.int(lim.max_models_per_job)} models per run.` : '';
    return `<section aria-labelledby="step-heading"><h2 id="step-heading" tabindex="-1">3. Models and policy</h2><p class="section-intro">Choose what to compare. Heavier recipes take longer; start small if you are unsure.</p>
      <div class="card"><fieldset id="model-list"${describe(state, 'model-list', false)}><legend>Models</legend>${errFor(state, 'model-list') ? `<p class="field-error" id="model-list-error">${esc(errFor(state, 'model-list').message)}</p>` : ''}${list}</fieldset><p class="notice" role="note">${esc(limitText)}</p></div>
      <div class="card"><h3>Policy</h3><p class="hint">The decision threshold of every model is chosen on the validation partition. You do not set it by hand.</p>${calib}${budgets}${costs}</div>
      <div class="actions step-actions">${button('back', 'Back', 'btn')}${button('next', 'Continue to run', 'btn btn-primary')}</div></section>`;
  }

  /* ---- step 4: run ---- */
  function runSummary(state) {
    const t = state.target, m = state.models, lim = state.data.limits || {};
    const kept = Object.keys(t.roles).filter(c => t.roles[c] !== 'include' && c !== t.column);
    return `<dl class="facts"><div><dt>Dataset</dt><dd>${esc(state.data.source ? (state.data.source.label || state.data.source.name) : '')}, ${U.int(state.data.profile.rows)} rows</dd></div><div><dt>Target</dt><dd>${esc(t.column)}; positive class ${esc(t.positive)}${t.oneVsRest ? ' against all other labels' : `, negative class ${esc(t.negative)}`}</dd></div><div><dt>Split</dt><dd>${esc((C.SPLIT_MODES.find(x => x.id === t.splitMode) || {}).name)}, seed ${esc(t.seed)}</dd></div><div><dt>Columns left out</dt><dd>${kept.length ? U.listText(kept) : 'none'}</dd></div><div><dt>Models</dt><dd>${U.listText(m.selected.map(modelName))}</dd></div><div><dt>Calibration</dt><dd>${esc((C.CALIBRATIONS.find(c => c.id === m.calibration) || {}).name)}</dd></div><div><dt>Review budgets</dt><dd>${esc(m.reviewPercents.trim() || 'none')}</dd></div><div><dt>Time limits</dt><dd>${lim.per_model_timeout_seconds ? `${U.duration(lim.per_model_timeout_seconds)} per model, ${U.duration(lim.job_timeout_seconds)} per run` : U.UNAVAILABLE}</dd></div></dl>`;
  }
  function renderRunLive(state) {
    const r = state.run, sel = r.selected || [];
    const doneIds = new Set((r.completed || []).map(c => c.model));
    const rows = sel.map(id => {
      const c = (r.completed || []).find(x => x.model === id);
      const text = c ? (c.status === 'succeeded' ? 'finished with a result' : String(c.status).replace(/_/g, ' ')) : (r.current === id ? 'fitting now' : 'waiting');
      return `<li><strong>${esc(modelName(id))}</strong>: ${esc(text)}</li>`;
    }).join('');
    const lines = [];
    lines.push(`<dt>State</dt><dd>${esc(r.state || U.UNAVAILABLE)}</dd>`);
    if (r.state === 'queued') lines.push(`<dt>Queue</dt><dd>${r.queuePosition != null ? `Position ${esc(r.queuePosition)}` : 'Waiting'}. The service runs one analysis at a time.</dd>`);
    if (r.state === 'running') lines.push(`<dt>Stage</dt><dd>${esc(C.STAGE_TEXT[r.stage] || r.stage || 'Working')}${r.current ? ` (${esc(modelName(r.current))})` : ''}</dd>`);
    lines.push(`<dt>Models finished</dt><dd>${U.int(doneIds.size)} of ${U.int(sel.length)}</dd>`);
    lines.push(`<dt>Elapsed at last check</dt><dd>${r.elapsed == null ? U.UNAVAILABLE : esc(U.seconds(r.elapsed))}</dd>`);
    if (r.limits) lines.push(`<dt>Limits</dt><dd>${esc(U.duration(r.limits.model))} per model, ${esc(U.duration(r.limits.job))} per run</dd>`);
    return `<dl class="facts run-facts">${lines.join('')}</dl>${rows ? `<ul class="model-progress">${rows}</ul>` : ''}<p class="muted">This shows events reported by the service: which stage it is in and which models have finished. It is not a percentage and not an estimate. The page checks about every ${esc((FLOW.POLL_MS / 1000).toFixed(1))} seconds.</p>`;
  }
  function endedBlock(state) {
    const r = state.run, st = r.state;
    const text = st === 'lost' ? 'The service no longer knows this analysis.' : (C.JOB_STATE_TEXT[st] || 'The analysis ended.');
    const reason = r.reason ? `<p><strong>Reason:</strong> ${esc(r.reason.code)}${C.REASONS[r.reason.code] ? ' - ' + esc(C.REASONS[r.reason.code]) : ''}</p>${r.reason.message ? `<p class="muted">Service message: ${esc(r.reason.message)}</p>` : ''}` : '';
    const models = (r.completed || []).length ? `<p>Models that finished before it ended:</p><ul>${r.completed.map(c => `<li>${esc(modelName(c.model))}: ${esc(c.status)}</li>`).join('')}</ul>` : '';
    const again = (st === 'expired' || st === 'lost' || st === 'interrupted') ? `${button('goto-data', 'Upload the dataset again', 'btn btn-primary')}${button('settings', 'Change settings', 'btn')}` : `${button('settings', 'Change settings and run again', 'btn btn-primary')}${button('goto-data', 'Use a different dataset', 'btn')}`;
    return `<div class="callout callout-warn" role="group" aria-labelledby="ended-title"><h3 id="ended-title" tabindex="-1">${esc(text)}</h3>${reason}${models}${r.notice ? `<p>${esc(r.notice)}</p>` : ''}<div class="actions">${again}</div></div>`;
  }
  function renderRun(state) {
    const r = state.run;
    let body;
    if (r.phase === 'idle') body = `<div class="card"><h3>Ready to run</h3>${runSummary(state)}<p class="notice" role="note">The file is already on the service. Starting the analysis fits the models on the service and uses its single analysis slot. If another analysis is running, you will see a busy message and can try again. Nothing starts until you press the button.</p><div class="actions">${button('back', 'Back', 'btn')}<button type="button" class="btn btn-primary" id="start-run" data-action="start">Start analysis</button></div></div>`;
    else if (r.phase === 'starting') body = `<div class="card"><h3>Starting</h3><p>Sending the settings to the service.</p></div>`;
    else if (r.phase === 'live' || r.phase === 'cancelling') body = `<div class="card"><h3 id="run-title" tabindex="-1">${r.phase === 'cancelling' ? 'Cancelling' : 'Analysis in progress'}</h3><div id="run-live">${renderRunLive(state)}</div>${r.contactLost ? `<p class="notice" role="note">${esc(r.notice)} ${button('recheck', 'Check again', 'btn')}</p>` : ''}<div class="actions"><button type="button" class="btn btn-danger" id="cancel-run" data-action="cancel"${r.phase === 'cancelling' ? ' disabled' : ''}>${r.phase === 'cancelling' ? 'Cancelling' : 'Cancel analysis'}</button></div><p class="hint">Cancel stops the worker on the service and keeps nothing from this run.</p></div>`;
    else body = `<div class="card">${endedBlock(state)}</div>`;
    return `<section aria-labelledby="step-heading"><h2 id="step-heading" tabindex="-1">4. Run</h2>${body}${r.phase === 'idle' ? renderPrivacyHosted(state) : ''}</section>`;
  }

  /* ---- step 5: results and export ---- */
  function renderDownloads(state) {
    const items = C.EXPORTS.map(x => `<li class="download-item"><div><strong>${esc(x.title)}</strong><p>${esc(x.text)}</p></div>${button('download', 'Download', 'btn', ` data-kind="${esc(x.kind)}" aria-label="Download ${esc(x.title)}"`)}</li>`).join('');
    return `<section class="card" aria-labelledby="dl-heading"><h3 id="dl-heading">Downloads</h3><p class="notice" role="note"><strong>No row-level data.</strong> ${esc(C.NO_ROW_DATA)}</p><ul class="downloads">${items}</ul><p class="hint">Row mappings, split indices and per-case scores are not offered in this version.</p><p id="download-status" class="status-line" role="status"></p></section>`;
  }
  function renderResults(state) {
    const res = state.result;
    if (!res) return `<section aria-labelledby="step-heading"><h2 id="step-heading" tabindex="-1">5. Results and export</h2><p>No result yet. Start an analysis in step 4.</p></section>`;
    const partial = res.state === 'partial';
    const l = state.data.limits;
    return `<section aria-labelledby="step-heading"><h2 id="step-heading" tabindex="-1">5. Results and export</h2>${partial ? '<p class="notice" role="note">Partial result: at least one model did not produce a result. It is listed with its reason code beside the models that did.</p>' : ''}
      ${RV.renderTrainingReport(res.report, { provenance: 'fresh_app_run', idPrefix: 'res', headingLevel: 3 })}
      ${renderDownloads(state)}
      <section class="card" aria-labelledby="priv-heading"><h3 id="priv-heading">Privacy and your uploaded data</h3><p>${esc(l ? C.PRIVACY.hostedAfter({ datasetIdle: U.duration(l.dataset_idle_seconds), finishedJob: U.duration(l.finished_job_seconds) }) : C.PRIVACY.hostedBefore)}</p><div class="actions">${button('delete-ask', 'Delete my uploaded data', 'btn btn-quiet')}${button('settings', 'Change settings and run again', 'btn')}${button('goto-data', 'Use a different dataset', 'btn')}</div>${state.ui.confirmDelete ? renderDeleteConfirm() : ''}</section></section>`;
  }

  function renderAnalyze(state, flow) {
    const body = state.step === 'data' ? renderData(state) : state.step === 'target' ? renderTarget(state) : state.step === 'models' ? renderModels(state) : state.step === 'run' ? renderRun(state) : renderResults(state);
    return `${renderStepper(state, flow)}${errorSummary(state)}<div class="step-body" id="step-body">${body}</div>`;
  }

  /* Analyze panel when no hosted service is present: a static page or the standalone local server. */
  function renderAnalyzeUnavailable() {
    return `<section aria-labelledby="analyze-heading"><h2 id="analyze-heading" tabindex="-1">Analyze a dataset</h2><p class="section-intro">Fitting models needs a Python backend. This page is running without one, so training is not available here.</p><div class="callout"><strong>Run the workbench on your own machine</strong><code>python -m rare_event_eval serve</code><p>Then open http://127.0.0.1:8601. Install the package with its training dependencies first; see the repository README. Each model has a 60-second worker limit and only one training request runs at a time.</p></div><p>You can still <a href="#/example" data-action="route" data-route="example">explore an example report</a> or <a href="#/evaluate" data-action="route" data-route="evaluate">evaluate predictions you already have</a>, both in this page.</p><aside class="privacy" aria-label="Privacy in this mode"><h3>Privacy</h3><p>${esc(C.PRIVACY.browser)}</p></aside></section>`;
  }

  /* ---- overview ---- */
  function renderOverview(env) {
    const caps = [
      ['Compare models', 'Fit several models on the same train, validation and test partitions, then read ranking quality, threshold outcomes and time side by side. A model that fails is shown beside the ones that worked.'],
      ['Examine evaluation choices', 'Choose what the model should generalize to: a random case, a new feature pattern, a new group or a later period. The threshold for each model is chosen on validation data only.'],
      ['Export evidence', 'Download a standalone report, structured JSON, the run settings and a reproducibility bundle. None of them contain row-level data.']
    ].map(([t, x]) => `<li class="capability"><h3>${esc(t)}</h3><p>${esc(x)}</p></li>`).join('');
    return `<section class="hero" aria-labelledby="overview-heading"><div class="hero-copy"><p class="eyebrow">Sherwan Ali / Applied machine learning</p><h1 id="overview-heading" tabindex="-1">FixYourDataset</h1><p class="lead">Evaluate imbalanced classification: compare models, examine resampling and split choices, select an operating policy on validation data, and export an inspectable report.</p><div class="actions"><a class="btn btn-primary" href="#/example" data-action="route" data-route="example">Explore example</a><a class="btn" href="#/analyze" data-action="route" data-route="analyze">Analyze your dataset</a></div>${env.hosted ? '' : '<p class="hint">Fitting on your own data needs the hosted service or a local Python install. Examples and prediction evaluation work in this page.</p>'}</div><div class="hero-preview" aria-label="Report preview">${env.preview || ''}<p class="hint">A preview of the report format. Open the example to read it in full.</p></div></section>
      <section aria-labelledby="cap-heading"><h2 id="cap-heading">What you can do</h2><ul class="capabilities">${caps}</ul></section>
      <section class="link-row" aria-label="Source, installation and research"><a href="${esc(C.SOURCE_URL)}" rel="noreferrer">Source code and saved results</a><span class="muted">Install and run: section 3 of the repository README</span><a href="#/research" data-action="route" data-route="research">Research evidence</a><a href="#/how" data-action="route" data-route="how">How it works</a></section>
      <section id="example-section" aria-labelledby="example-heading"><h2 id="example-heading" tabindex="-1">Explore an example</h2><div id="example-root"></div></section>`;
  }

  /* ---- research ---- */
  function renderResearch(study) {
    const R = C.RESEARCH, n = R.november;
    const prov = RV.badge('historical_research_record');
    const rows = n.rows.map(r => [r.against, U.int(r.higher), U.int(r.tied), U.int(r.lower), r.difference, r.note]);
    const pilotRows = study ? study.family_comparisons.map(r => [r.dataset, U.fixed(r.erf.mean_ap, 4), U.fixed(r.balanced_rf.mean_ap, 4), U.fixed(r.rf_weighted_1000.mean_ap, 4), U.fixed(r.hist_gradient_boosting.mean_ap, 4), U.signed(r.erf_minus_balanced_rf_ap, 4)]) : [];
    return `<section aria-labelledby="research-heading"><h2 id="research-heading" tabindex="-1">Research</h2>${prov}<p class="lead-sm">${esc(R.notPeerReviewed)} ${esc(R.appVersusResearch)}</p>
      <div class="card"><h3>ERF-CSRBoost in brief</h3><p>${esc(R.whatIsERF)}</p></div>
      <div class="card"><h3>Latest comparison: November 2026 record</h3><p>${esc(n.headline)}</p><p>${esc(n.extension)}</p>${RV.table(RV.makeIds('rs'), 'ERF against three baselines on mean average precision, counts are numbers of datasets', [{ text: 'ERF against' }, { text: 'Higher on', num: true }, { text: 'Tied on', num: true }, { text: 'Lower on', num: true }, { text: 'Mean AP difference', num: true }, { text: 'Note' }], rows)}<p class="table-note">Counts are numbers of datasets. They are not significance tests, and no p-value is given for the 20 datasets.</p>
        <h4>What limits these numbers</h4><ul>${n.limits.map(x => `<li>${esc(x)}</li>`).join('')}</ul><p>${esc(R.reproducibility)} <a href="${esc(C.NOVEMBER_RECORD_URL)}" rel="noreferrer">The November record</a>.</p></div>
      <div class="card"><h3>Controlled pilot (separate protocol)</h3>${RV.badge('historical_research_record')}<p>${esc(R.pilot.intro)}</p><ul>${R.pilot.points.map(x => `<li>${esc(x)}</li>`).join('')}</ul><p class="notice">${esc(R.pilot.caution)}</p>${pilotRows.length ? RV.table(RV.makeIds('pl'), 'Pilot: mean average precision of two stratified splits per dataset family', [{ text: 'Family' }, { text: 'ERF', num: true }, { text: 'Balanced RF', num: true }, { text: 'RF 1,000 trees', num: true }, { text: 'Gradient boosting', num: true }, { text: 'ERF minus balanced RF', num: true }], pilotRows) : ''}<figure class="study-figure"><img src="quality-vs-cost.png" alt="Five exploratory dataset comparisons of average precision and model fit time. ERF did not meet the predeclared expansion rule." loading="lazy"><figcaption>Pilot: two split means per family, with shared-workstation timings. Families have different prevalence, so average precision is not comparable across them.</figcaption></figure></div>
      <div class="card"><h3>Adaptive campaign (separate procedure)</h3>${RV.badge('historical_research_record')}<p>${esc(R.adaptive)}</p></div>
      <div class="card"><h3>The CSRBoost audit (separate project)</h3><p>${esc(R.audit.text)}</p><p><a href="${esc(C.AUDIT_URL)}" rel="noreferrer">Audit repository</a></p></div>
      <p class="link-row"><a href="${esc(C.SOURCE_URL)}" rel="noreferrer">Source code and saved results</a><a href="${esc(C.AUDIT_URL)}" rel="noreferrer">Audit repository</a><a href="#/how" data-action="route" data-route="how">How this app evaluates</a></p></section>`;
  }

  /* ---- how it works ---- */
  function renderHow() {
    const H = C.HOW, ul = items => `<ul>${items.map(x => `<li>${esc(x)}</li>`).join('')}</ul>`;
    const glossary = Object.keys(C.GLOSSARY).map(k => `<div><dt>${esc(C.GLOSSARY[k].term)}</dt><dd>${esc(C.GLOSSARY[k].text)}</dd></div>`).join('');
    const modes = C.SPLIT_MODES.map(m => `<div><dt>${esc(m.name)}</dt><dd>${esc(m.question)} ${esc(m.detail)}</dd></div>`).join('');
    return `<section aria-labelledby="how-heading"><h2 id="how-heading" tabindex="-1">How it works</h2>
      <div class="card"><h3>What happens in an analysis</h3><ol class="how-steps">${H.steps.map(s => `<li><strong>${esc(s.title)}.</strong> ${esc(s.text)}</li>`).join('')}</ol></div>
      <div class="card"><h3>Split modes</h3><dl class="facts">${modes}</dl></div>
      <div class="card"><h3>The leakage boundary</h3>${ul(H.leakage)}</div>
      <div class="card"><h3>Recipe differences from the older app</h3>${ul(H.recipes)}<p class="hint">The full comparison is in docs/RECIPE-DIFFERENCES.md of the repository.</p></div>
      <div class="card"><h3>Failure policy and limits</h3>${ul(H.failure)}</div>
      <div class="card"><h3>What a report does not establish</h3>${ul(H.notEstablished)}</div>
      <div class="card"><h3>Terms</h3><dl class="facts">${glossary}</dl></div>
      <div class="card"><h3>Older tools</h3><p>The earlier single-page app is kept at <a href="/classic" data-classic="1">Classic tools</a> when this page is served by the hosted service. It holds the leakage demonstration, scoring of new rows, the balanced training and held-out file exports and the honesty receipt, which are not part of this workbench yet.</p></div></section>`;
  }

  /* ---- Evaluate predictions: text blocks the page markup needs ---- */
  function renderPredictionNotes() {
    return { privacy: C.PRIVACY.browser, cannot: root.FYD.predictions.CANNOT_CERTIFY };
  }

  /* ---- the notes that come with the recorded example (all text is escaped) ---- */
  function renderRecordedInfo(example) {
    if (!example || typeof example !== 'object') return '';
    const g = example.generator && typeof example.generator === 'object' ? example.generator : {};
    const quote = a => (/\s/.test(String(a)) ? `"${a}"` : String(a));
    const replay = Array.isArray(g.replay_command_arguments) && g.replay_command_arguments.length ? `<details class="more"><summary>Command-line replay of the same settings</summary><pre class="code-block" tabindex="0" aria-label="Replay command">${esc(g.replay_command_arguments.map(quote).join(' '))}</pre><p class="hint">Run it from the repository root with the package installed. The aggregate metrics should match; timings and the creation time differ.</p></details>` : '';
    return `<section class="card" aria-labelledby="about-example"><h3 id="about-example">About this recorded run</h3>${example.title ? `<p><strong>${esc(example.title)}</strong></p>` : ''}${example.description ? `<p>${esc(example.description)}</p>` : ''}${example.data_origin ? `<p><strong>Data origin.</strong> ${esc(example.data_origin)}</p>` : ''}${example.note ? `<p class="notice" role="note">${esc(example.note)}</p>` : ''}${g.regenerate_command ? `<p>The generator and the run that produced it are in the repository. To regenerate it:</p><pre class="code-block" tabindex="0" aria-label="Regenerate command">${esc(g.regenerate_command)}</pre>` : ''}${replay}</section>`;
  }

  /* ---- Example controls ---- */
  function renderExampleChooser(examples, selectedId) {
    return `<fieldset class="example-list"><legend>Choose an example</legend>${examples.map(e => `<div class="radio-row"><input type="radio" name="example" id="example-${esc(e.id)}" value="${esc(e.id)}" data-action="pick-example"${e.id === selectedId ? ' checked' : ''}${e.available === false ? ' disabled' : ''}><label for="example-${esc(e.id)}"><strong>${esc(e.title)}</strong> <span class="pill">${esc(C.PROVENANCE[e.provenance].label)}</span><span class="radio-desc">${esc(e.description)}</span></label></div>`).join('')}</fieldset>`;
  }

  const api = { renderAnalyze, renderAnalyzeUnavailable, renderStepper, renderData, renderTarget, renderModels, renderRun, renderRunLive, renderResults, renderOverview, renderResearch, renderHow,
    renderPredictionNotes, renderExampleChooser, renderRecordedInfo, errorSummary, STEP_TITLES };
  root.FYD.views = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
