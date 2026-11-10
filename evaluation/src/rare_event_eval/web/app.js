/* Sherwan Ali. Browser glue for the workbench: navigation, event wiring and focus handling. The logic lives in the
   pure modules (flow.js, views.js, report-view.js, predictions.js); this file holds no network code, which is in
   api-client.js only. Access tokens never reach this file. */
(function () {
  'use strict';
  // A script that failed to load leaves a global missing. preflight.js then shows the alert with its Reload button.
  if (window.FYD_PREFLIGHT && window.FYD_PREFLIGHT.missing(window).length) return;
  const F = window.FYD, U = F.util, C = F.content, RV = F.reportView, V = F.views, P = F.predictions, MODELS = F.MODELS, M = window.RareEventMetrics, esc = U.esc;
  const $ = id => document.getElementById(id);
  const api = F.api.createApiClient({ pageSearch: location.search, pageReferrer: document.referrer });
  const flow = F.flow.createAnalysisFlow({ api });
  const env = { mode: 'static', hosted: false, recorded: null, version: null, detected: false };
  const TITLES = { overview: 'Overview', example: 'Explore an example', analyze: 'Analyze a dataset', tools: 'Data tools', evaluate: 'Evaluate predictions', research: 'Research', how: 'How it works' };
  const ROUTES = { overview: 'overview', example: 'overview', analyze: 'analyze', tools: 'analyze', evaluate: 'evaluate', research: 'research', how: 'how' };
  const TABS = ['overview', 'analyze', 'evaluate', 'research', 'how'];
  let currentRoute = null;
  /* The live region is cleared first and filled a moment later, so the same message can be announced again and a change made together with a redraw is not lost. */
  let sayTimer = null;
  const say = text => {
    const el = $('live-status');
    if (!el) return;
    el.textContent = '';
    if (sayTimer) clearTimeout(sayTimer);
    sayTimer = setTimeout(() => { el.textContent = text; sayTimer = null; }, 60);
  };

  /* ---------- routing and tabs ---------- */
  function routeFromHash() {
    const m = /^#\/([a-z]+)/.exec(location.hash);
    return m && ROUTES[m[1]] ? m[1] : null;
  }
  function show(route, options = {}) {
    const tab = ROUTES[route];
    TABS.forEach(name => {
      const selected = name === tab, button = $('tab-' + name);
      button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
      $('panel-' + name).hidden = !selected;
    });
    document.title = `${TITLES[route] || TITLES[tab]} | FixYourDataset`;
    if (currentRoute !== route) say(`${TITLES[route] || TITLES[tab]} page.`);
    if (currentRoute === 'tools' && route !== 'tools') {
      tools.reset(); $('data-tools-root').innerHTML = '';
    }
    currentRoute = route;
    $('analyze-root').hidden = route === 'tools';
    $('data-tools-root').hidden = route !== 'tools';
    if (route === 'tools') {
      tools.reset(); $('data-tools-root').innerHTML = F.tools.render(env.hosted);
    } else if (tab === 'analyze') renderAnalyzePanel(options);
    if (route === 'example') { const h = $('example-heading'); if (h && options.focus) { h.scrollIntoView(); h.focus({ preventScroll: true }); } }
    else if (options.focus) { const h = $('panel-' + tab).querySelector('h1,h2'); if (h) { window.scrollTo(0, 0); h.focus({ preventScroll: true }); } }
  }
  function go(route, focus) {
    const target = '#/' + route;
    if (location.hash === target) show(route, { focus }); else { pendingFocus = !!focus; location.hash = target; }
  }
  let pendingFocus = false;
  window.addEventListener('hashchange', () => { const r = routeFromHash(); if (!r) return; show(r, { focus: pendingFocus }); pendingFocus = false; });
  document.querySelectorAll('[data-tab]').forEach(button => {
    button.addEventListener('click', () => go(button.dataset.tab, false));
    button.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const i = TABS.indexOf(button.dataset.tab);
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? TABS.length - 1 : (i + (event.key === 'ArrowRight' ? 1 : -1) + TABS.length) % TABS.length;
      go(TABS[next], false); $('tab-' + TABS[next]).focus();
    });
  });
  document.addEventListener('click', event => {
    const link = event.target.closest('[data-action="route"]');
    if (link) { event.preventDefault(); go(link.dataset.route, true); }
  });

  /* ---------- downloads ---------- */
  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const saveText = (text, name, type) => saveBlob(new Blob([text], { type }), name);
  const tools = F.tools.attach({ api, save: saveText, announce: say });

  /* ---------- Overview and example ---------- */
  const FICTIONAL = window.ERF_EXAMPLES;
  const ex = { id: null, picked: false, budget: 2, threshold: 0.5, view: 'budget' };
  function fictionalReport(item, budget, threshold) {
    const parsed = M.fromCSV(item.csv, 'outcome', ['baseline', 'candidate']);
    return { report: M.evaluate(parsed.labels, parsed.scores, { positiveLabel: 'event', negativeLabel: 'clear', scoreKind: 'probability', reviewBudgets: [budget], threshold, description: item.description }), rows: parsed.labels.length };
  }
  function exampleList() {
    const list = [{ id: 'recorded', title: 'Recorded app run', provenance: 'recorded_app_run', available: !!env.recorded,
      description: env.recorded ? ((env.recorded.example && env.recorded.example.description) || 'A real run of this app on a seeded synthetic table: several models on the same held-out test cases, with validation-selected thresholds.') : 'Not available in this copy of the page. It is loaded from recorded-example.json when the page is served over http.' }];
    FICTIONAL.forEach(item => list.push({ id: item.id, title: item.title, provenance: 'fictional_fixture', description: item.description }));
    return list;
  }
  function renderExample() {
    const root = $('example-root');
    if (!root) return;
    const list = exampleList();
    if (!ex.picked || !list.some(e => e.id === ex.id && e.available !== false)) ex.id = list[0].available === false ? FICTIONAL[0].id : 'recorded';
    const chosen = list.find(e => e.id === ex.id);
    const fictional = ex.id !== 'recorded';
    const links = `<p class="link-row"><a href="${esc(C.SOURCE_URL)}" rel="noreferrer">Code and saved results</a><a href="#/how" data-action="route" data-route="how">Method</a><a href="#/research" data-action="route" data-route="research">Research evidence</a>${!fictional && env.recorded ? '<a href="recorded-example.html">Open the standalone report page</a>' : ''}</p>`;
    const controls = fictional ? `<div class="card example-controls"><div class="field"><label for="review-budget">Cases you can review: <span id="budget-label">${esc(ex.budget)}</span></label><input id="review-budget" type="range" min="0" max="10" value="${esc(ex.budget)}" aria-describedby="budget-hint"><p class="hint" id="budget-hint">A count of cases, not a false-positive rate.</p></div><div class="field"><label for="example-threshold">Exploratory threshold: <span id="threshold-label">${esc(ex.threshold.toFixed(2))}</span></label><input id="example-threshold" type="range" min="0" max="1" step="0.01" value="${esc(ex.threshold)}" aria-describedby="threshold-hint"><p class="hint" id="threshold-hint">Changing this after looking at outcomes is exploration. A final threshold needs independent validation.</p></div><div class="field"><label for="decision-view">Decision shown in the table</label><select id="decision-view"><option value="budget"${ex.view === 'budget' ? ' selected' : ''}>Review budget</option><option value="threshold"${ex.view === 'threshold' ? ' selected' : ''}>Fixed threshold</option></select></div></div>` : '';
    root.innerHTML = `<p class="section-intro">Read a complete report without uploading anything or waiting for a model. Each example says what it is.</p>${V.renderExampleChooser(list, ex.id)}${!fictional && env.recorded ? V.renderRecordedInfo(env.recorded.example) : ''}${controls}<div id="example-report" aria-label="${esc(chosen.title)}"></div>${links}<aside class="privacy" aria-label="Privacy in this mode"><h3>Privacy</h3><p>${esc(C.PRIVACY.example)}</p></aside>`;
    renderExampleReport();
  }
  function renderExampleReport() {
    const holder = $('example-report');
    if (!holder) return;
    if (ex.id === 'recorded') { holder.innerHTML = env.recorded ? RV.renderTrainingReport(env.recorded, { provenance: 'recorded_app_run', idPrefix: 'ex', headingLevel: 3 }) : ''; return; }
    const item = FICTIONAL.find(e => e.id === ex.id), r = fictionalReport(item, ex.budget, ex.threshold);
    const range = $('review-budget');
    if (range) { range.max = String(r.rows); if (ex.budget > r.rows) { ex.budget = r.rows; range.value = String(r.rows); } }
    $('budget-label') && ($('budget-label').textContent = String(ex.budget));
    $('threshold-label') && ($('threshold-label').textContent = ex.threshold.toFixed(2));
    holder.innerHTML = RV.renderPredictionReport(r.report, { provenance: 'fictional_fixture', view: ex.view, idPrefix: 'ex' });
  }
  document.addEventListener('change', event => {
    const t = event.target;
    if (t.matches && t.matches('[data-action="pick-example"]')) { ex.id = t.value; ex.picked = true; renderExample(); }
    if (t.id === 'decision-view') { ex.view = t.value; renderExampleReport(); }
  });
  document.addEventListener('input', event => {
    const t = event.target;
    if (t.id === 'review-budget') { ex.budget = +t.value; renderExampleReport(); }
    if (t.id === 'example-threshold') { ex.threshold = +t.value; renderExampleReport(); }
  });
  function renderOverview() {
    const first = FICTIONAL[0];
    const preview = env.recorded ? RV.renderPreview(env.recorded, { provenance: 'recorded_app_run' }) : RV.renderPreview(fictionalReport(first, 2, 0.5).report, { provenance: 'fictional_fixture' });
    $('panel-overview').innerHTML = V.renderOverview({ hosted: env.hosted, preview });
    renderExample();
  }

  /* ---------- Analyze: hosted flow ---------- */
  let lastStep = flow.getState().step, lastErrorsVersion = 0, lastStatusVersion = 0, lastFocusN = 0, lastRunKey = '', analyzeInit = false, localInit = false;
  function renderAnalyzePanel() {
    const root = $('analyze-root');
    if (!env.detected) { root.innerHTML = '<p class="muted">Checking what this page can connect to.</p>'; return; }
    if (env.mode === 'hosted') {
      if (!analyzeInit) { analyzeInit = true; flow.loadSamples(); }
      drawFlow(true);
    } else if (env.mode === 'local') { if (!localInit) { localInit = true; root.innerHTML = localTrainingHtml(); } }
    else root.innerHTML = V.renderAnalyzeUnavailable();
  }
  function drawFlow(force) {
    const st = flow.getState(), root = $('analyze-root');
    if (env.mode !== 'hosted' || !root || $('panel-analyze').hidden) { return; }
    const key = [st.step, st.run.phase, st.errorsVersion, st.data.phase, !!st.result, st.ui.confirmDelete, st.run.contactLost, st.samples.phase, st.run.state].join('|');
    if (!force && st.step === 'run' && key === lastRunKey && $('run-live')) { $('run-live').innerHTML = V.renderRunLive(st); return; }
    lastRunKey = key;
    if (force) { lastStep = st.step; lastErrorsVersion = st.errorsVersion; lastFocusN = st.focus.n; }
    const active = document.activeElement, activeId = active && root.contains(active) ? active.id : '';
    root.innerHTML = V.renderAnalyze(st, flow);
    if (st.errors.length && st.errorsVersion !== lastErrorsVersion && $('error-summary')) { $('error-summary').focus(); }
    else if (st.focus.n !== lastFocusN && st.focus.id && $(st.focus.id)) { $(st.focus.id).focus(); }
    else if (st.step !== lastStep && $('step-heading')) { $('step-heading').focus({ preventScroll: false }); }
    else if (activeId && $(activeId)) { $(activeId).focus({ preventScroll: true }); }
    lastStep = st.step; lastErrorsVersion = st.errorsVersion; lastFocusN = st.focus.n;
  }
  flow.subscribe(st => {
    if (st.statusVersion !== lastStatusVersion) { lastStatusVersion = st.statusVersion; say(st.status); }
    drawFlow(false);
  });
  async function startDownload(kind) {
    const out = $('download-status');
    if (out) out.textContent = 'Preparing the download.';
    try {
      const { blob, name } = await flow.download(kind);
      saveBlob(blob, name);
      if ($('download-status')) $('download-status').textContent = U.markBidi(`Saved ${name}.`);
    } catch (e) {
      const text = e && e.code === 'EXPORT_UNKNOWN' ? 'This service does not offer that download.' : F.flow.describeError(e) || 'The download did not work.';
      if ($('download-status')) $('download-status').textContent = text;
    }
  }
  document.addEventListener('click', event => {
    const el = event.target.closest('#analyze-root [data-action]');
    if (!el || env.mode !== 'hosted') return;
    const a = el.dataset.action;
    if (a === 'focus-field') {
      event.preventDefault();
      const id = el.dataset.field;
      const target = $(id) || document.querySelector(`#${CSS.escape(id)} input, [name="${CSS.escape(id)}"]`) || $('step-heading');
      if (target) { target.focus(); target.scrollIntoView({ block: 'center' }); }
      return;
    }
    if (a === 'upload') flow.uploadSelected();
    else if (a === 'retry-samples') flow.retrySamples();
    else if (a === 'next') flow.next();
    else if (a === 'back') flow.back();
    else if (a === 'goto') flow.goTo(el.dataset.step);
    else if (a === 'start') flow.startRun();
    else if (a === 'cancel') flow.cancel();
    else if (a === 'recheck') flow.recheck();
    else if (a === 'settings') flow.backToSettings();
    else if (a === 'goto-data') flow.goTo('data');
    else if (a === 'download') startDownload(el.dataset.kind);
    else if (a === 'delete-ask') flow.setConfirmDelete(true);
    else if (a === 'delete-cancel') flow.setConfirmDelete(false);
    else if (a === 'delete-confirm') flow.discardDataset();
  });
  document.addEventListener('change', event => {
    const el = event.target;
    if (!el.closest || !el.closest('#analyze-root') || env.mode !== 'hosted') return;
    const a = el.dataset && el.dataset.action;
    if (el.id === 'file-input') { if (el.files && el.files[0]) flow.selectFile(el.files[0]); return; }
    if (a === 'import-selection') flow.setImportSelection(el.dataset.kind,el.value);
    else if (a === 'pick-sample') flow.selectSample(el.value);
    else if (a === 'set-target') flow.setTargetColumn(el.value);
    else if (el.id === 'positive-label' && el.tagName === 'SELECT') flow.setPositive(el.value);
    else if (a === 'set-ovr') flow.setField('target', 'oneVsRest', el.checked);
    else if (a === 'set-role') flow.setColumnRole(el.dataset.column, el.value);
    else if (a === 'set-split') flow.setSplitMode(el.value);
    else if (a === 'set-missing') flow.setField('target', 'missingPolicy', el.value);
    else if (a === 'confirm-target') flow.setField('target', 'confirmed', el.checked);
    else if (a === 'toggle-model') flow.toggleModel(el.value, el.checked);
    else if (a === 'set-calibration') flow.setField('models', 'calibration', el.value);
    else if (el.id === 'group-column') flow.setField('target', 'groupColumn', el.value);
    else if (el.id === 'time-column') flow.setField('target', 'timeColumn', el.value);
  });
  document.addEventListener('input', event => {
    const el = event.target;
    if (!el.closest || !el.closest('#analyze-root') || env.mode !== 'hosted') return;
    const resetConfirm = () => { const c = $('confirm-target'); if (c) c.checked = false; };
    if (el.id === 'positive-label' && el.tagName === 'INPUT') { flow.setField('target', 'positive', el.value, true); resetConfirm(); }
    else if (el.id === 'negative-label') { flow.setField('target', 'negative', el.value, true); resetConfirm(); }
    else if (el.id === 'seed') { flow.setField('target', 'seed', el.value, true); resetConfirm(); }
    else if (el.id === 'review-budgets') flow.setField('models', 'reviewPercents', el.value, true);
    else if (el.dataset && el.dataset.action === 'set-cost') flow.setCost(el.dataset.key, el.value);
  });
  document.addEventListener('dragover', event => { const z = event.target.closest && event.target.closest('#dropzone'); if (z) { event.preventDefault(); z.classList.add('is-over'); } });
  document.addEventListener('dragleave', event => { const z = event.target.closest && event.target.closest('#dropzone'); if (z) z.classList.remove('is-over'); });
  document.addEventListener('drop', event => {
    const z = event.target.closest && event.target.closest('#dropzone');
    if (!z) return;
    event.preventDefault(); z.classList.remove('is-over');
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) flow.selectFile(file);
  });

  /* ---------- Analyze: standalone local server (the older numeric-only form) ---------- */
  const local = { csv: '', headers: [] };
  function localTrainingHtml() {
    const models = MODELS.map(m => `<div class="check-row"><input type="checkbox" id="local-model-${esc(m.id)}" name="local-model" value="${esc(m.id)}"${['logistic', 'rf_weighted'].includes(m.id) ? ' checked' : ''}><label for="local-model-${esc(m.id)}">${esc(m.name)}</label></div>`).join('');
    const splits = C.SPLIT_MODES.map(s => `<option value="${esc(s.id)}">${esc(s.name)}</option>`).join('');
    return `<section aria-labelledby="local-heading"><h2 id="local-heading" tabindex="-1">Analyze a dataset on this machine</h2><p class="section-intro">The local Python workbench is connected. Training uses the installed package and your CSV stays on this computer. Features must be numeric. Each model has a 60-second worker limit and only one training request runs at a time.</p>
      <div class="workspace"><form id="local-form" class="card" novalidate><div class="field"><label for="local-file">Numeric-feature CSV</label><input id="local-file" type="file" accept=".csv,text/csv"></div><button type="button" class="btn" id="local-example">Load fictional training cases</button><p class="hint" id="local-note">The fictional training cases are for software demonstration; they are not a real case study.</p>
      <div class="field"><label for="local-target">Target column</label><select id="local-target"><option value="">Load a file first</option></select></div><div class="field"><label for="local-positive">Positive label</label><input id="local-positive" type="text"></div><div class="field"><label for="local-negative">Negative label</label><input id="local-negative" type="text"></div>
      <div class="field"><label for="local-split">Generalization question</label><select id="local-split">${splits}</select></div><div class="field"><label for="local-group">Group column, if applicable</label><select id="local-group"><option value="">None</option></select></div><div class="field"><label for="local-time">Time column, if applicable</label><select id="local-time"><option value="">None</option></select></div>
      <fieldset><legend>Models</legend>${models}</fieldset><div class="field"><label for="local-calibration">Probability calibration</label><select id="local-calibration"><option value="none">None</option><option value="sigmoid">Sigmoid, training-only cross-validation</option></select></div>
      <button type="submit" class="btn btn-primary" id="local-run">Run local comparison</button><p id="local-status" class="status-line" role="status"></p></form><div id="local-results" aria-live="polite"><div class="empty-state"><h3>A traceable fresh run</h3><p>Training-only preprocessing, validation-only threshold selection, explicit failures and an untouched test partition. Results appear here.</p></div></div></div>
      <aside class="privacy" aria-label="Privacy in this mode"><h3>Privacy: local training</h3><p>The file is sent only to the Python process on your own machine at 127.0.0.1. Nothing is stored by it. The JSON download is built in your browser and holds no row-level data.</p></aside></section>`;
  }
  function localLoad(csv, fictional) {
    const parsed = M.parseCSV(csv); local.csv = csv; local.headers = parsed.headers;
    const opts = (blank) => (blank ? '<option value="">None</option>' : '') + parsed.headers.map(h => `<option value="${esc(h)}">${esc(h)}</option>`).join('');
    $('local-target').innerHTML = opts(false); $('local-group').innerHTML = opts(true); $('local-time').innerHTML = opts(true);
    if (fictional) { $('local-target').value = 'outcome'; $('local-positive').value = 'event'; $('local-negative').value = 'clear'; } else { $('local-positive').value = ''; $('local-negative').value = ''; }
    $('local-note').textContent = `${parsed.rows.length} rows loaded. ${fictional ? 'Fictional teaching data.' : 'Confirm which columns are available at prediction time.'}`;
  }
  document.addEventListener('click', event => {
    if (event.target.id === 'local-example') localLoad(window.ERF_TRAINING_CSV, true);
  });
  document.addEventListener('change', async event => {
    if (event.target.id !== 'local-file') return;
    const file = event.target.files[0]; if (!file) return;
    try { if (file.size > 2000000) throw Error('Please use a CSV under 2 MB in the workbench, or use the command line.'); localLoad(await file.text(), false); } catch (e) { $('local-status').textContent = e.message; }
  });
  document.addEventListener('submit', async event => {
    if (event.target.id !== 'local-form') return;
    event.preventDefault();
    const status = $('local-status'), button = $('local-run');
    if (!local.csv) { status.textContent = 'Load a training CSV first.'; return; }
    const models = [...document.querySelectorAll('input[name=local-model]:checked')].map(i => i.value);
    if (!models.length || models.length > 5) { status.textContent = 'Choose one to five models.'; return; }
    button.disabled = true; status.textContent = 'Running bounded workers on this machine. A model may take up to 60 seconds.';
    try {
      const data = await api.trainLocal({ csv: local.csv, target: $('local-target').value, positive_label: $('local-positive').value, negative_label: $('local-negative').value, models, split_mode: $('local-split').value, group_column: $('local-group').value || null, time_column: $('local-time').value || null, calibration: $('local-calibration').value, data_description: 'User-supplied numeric training data; provenance is not independently verified.' });
      $('local-results').innerHTML = RV.renderTrainingReport(data, { provenance: 'fresh_app_run', idPrefix: 'loc' }) + `<div class="actions"><button type="button" class="btn" id="local-json">Download report (JSON)</button></div><p class="hint">${esc(C.NO_ROW_DATA)}</p>`;
      $('local-json').onclick = () => saveText(JSON.stringify({ ...data, provenance: 'fresh app run' }, null, 2), U.downloadName('report_json', 'local-training', new Date().toISOString()), 'application/json');
      status.textContent = 'Finished. Inspect the validation-selected policies, the held-out results and any model failures.';
    } catch (e) { status.textContent = U.markBidi(e.message); } finally { button.disabled = false; }
  });

  /* ---------- Evaluate predictions: browser only ---------- */
  const pred = { csv: '', name: '', last: null, view: 'budget' };
  const PRED_NOTE = 'Computed in this browser from the predictions you supplied. How they were produced is not known to this app.';
  function predOptions(headers) { $('prediction-label').innerHTML = headers.map(h => `<option value="${esc(h)}">${esc(h)}</option>`).join(''); }
  function scoreChoices() {
    const headers = M.parseCSV(pred.csv).headers;
    $('score-columns').innerHTML = headers.filter(h => h !== $('prediction-label').value).map((h, i) => `<div class="check-row"><input type="checkbox" id="score-${i}" value="${esc(h)}"><label for="score-${i}">${esc(h)}</label></div>`).join('');
  }
  function loadPrediction(csv, name, fictional) {
    const parsed = M.parseCSV(csv); pred.csv = csv; pred.name = name; predOptions(parsed.headers); scoreChoices();
    $('prediction-positive').value = fictional ? 'event' : ''; $('prediction-negative').value = fictional ? 'clear' : '';
    if (fictional) { $('prediction-label').value = 'outcome'; scoreChoices(); $('score-columns').querySelectorAll('input').forEach(i => { i.checked = true; }); }
    $('prediction-status').textContent = U.markBidi(`Loaded ${parsed.rows.length} rows from ${name}. Confirm the labels, the score columns and the score direction.`);
    $('prediction-errors').innerHTML = '';
  }
  function showPredictionErrors(errors) {
    $('prediction-errors').innerHTML = V.errorSummary({ errors });
    const summary = $('error-summary'); if (summary) summary.focus();
    $('prediction-results').innerHTML = '<div class="empty-state"><h3>Input needs attention</h3><p>The previous result is cleared. Correct the items listed above and evaluate again.</p></div>';
    $('prediction-status').textContent = `${errors.length} ${errors.length === 1 ? 'problem needs' : 'problems need'} attention.`;
  }
  function drawPrediction() {
    const r = pred.last; if (!r) return;
    $('prediction-results').innerHTML = `<div class="actions"><label for="prediction-view" class="visually-hidden">Decision shown</label><select id="prediction-view"><option value="budget"${pred.view === 'budget' ? ' selected' : ''}>Decision shown: review budget</option><option value="threshold"${pred.view === 'threshold' ? ' selected' : ''}>Decision shown: fixed threshold</option></select></div>` +
      RV.renderPredictionReport(r, { provenance: 'fresh_app_run', provenanceNote: PRED_NOTE, view: pred.view, idPrefix: 'pr', cannotCertify: P.CANNOT_CERTIFY }) +
      `<div class="card"><h3>Downloads</h3><p>${esc(C.NO_ROW_DATA)}</p><div class="actions"><button type="button" class="btn" id="pred-json">Download report (JSON)</button><button type="button" class="btn" id="pred-html">Download report (HTML)</button></div></div>`;
  }
  $('use-prediction-example').addEventListener('click', () => loadPrediction(FICTIONAL[0].csv, 'the fictional example', true));
  $('prediction-label').addEventListener('change', () => { if (pred.csv) scoreChoices(); });
  $('prediction-file').addEventListener('change', async () => {
    const file = $('prediction-file').files[0]; if (!file) return;
    try { if (file.size > P.MAX_BYTES) throw Error('Please use a CSV under 2 MB in the workbench, or use the command line.'); loadPrediction(await file.text(), file.name, false); }
    catch (e) { showPredictionErrors([{ field: 'prediction-file', message: e.message }]); }
  });
  $('prediction-form').addEventListener('submit', event => {
    event.preventDefault();
    const form = { labelColumn: $('prediction-label').value, positiveLabel: $('prediction-positive').value, negativeLabel: $('prediction-negative').value,
      scoreColumns: [...$('score-columns').querySelectorAll('input:checked')].map(i => i.value), scoreKind: $('score-kind').value, threshold: $('prediction-threshold').value,
      budget: $('prediction-budget').value, costs: { fp: $('cost-fp').value, fn: $('cost-fn').value, review: $('cost-review').value } };
    const out = P.evaluateSaved(pred.csv, form);
    if (out.errors) { showPredictionErrors(out.errors); return; }
    pred.last = out.report; $('prediction-errors').innerHTML = ''; drawPrediction();
    $('prediction-status').textContent = 'Evaluated in your browser. No CSV was sent to a server. The report is below.';
  });
  document.addEventListener('change', event => { if (event.target.id === 'prediction-view') { pred.view = event.target.value; drawPrediction(); const s = $('prediction-view'); if (s) s.focus(); } });
  document.addEventListener('click', event => {
    const id = event.target.id;
    if ((id === 'pred-json' || id === 'pred-html') && pred.last) {
      const date = pred.last.created_utc;
      if (id === 'pred-json') saveText(P.reportJson(pred.last, C.PROVENANCE.fresh_app_run.label), U.downloadName('report_json', pred.name === 'the fictional example' ? 'fictional-predictions' : pred.name, date), 'application/json');
      else saveText(RV.predictionReportDocument(pred.last, { provenance: 'fresh_app_run', provenanceNote: PRED_NOTE, view: pred.view, cannotCertify: P.CANNOT_CERTIFY }), U.downloadName('report_html', pred.name === 'the fictional example' ? 'fictional-predictions' : pred.name, date), 'text/html');
    }
  });

  /* ---------- boot ---------- */
  function setBadge(text) { $('runtime-badge').textContent = text; }
  async function detect() {
    const forced = new URLSearchParams(location.search).get('mode') === 'browser';
    if (location.protocol === 'file:' || forced) { env.mode = 'static'; env.detected = true; setBadge('Browser only'); return; }
    const hosted = await api.detectHosted();
    if (hosted) {
      env.mode = 'hosted'; env.hosted = true; env.version = hosted; flow.setVersion(hosted);
      setBadge('Hosted analysis available'); $('classic-link').hidden = false;
      const rel = hosted.release_id && hosted.release_id !== 'unknown' ? ` Release ${hosted.release_id}.` : '';
      const fv = $('footer-version'); if (fv) fv.textContent = `Service app ${hosted.app_version || ''}, package ${hosted.package_version || ''}.${rel}`.replace(/\s+/g, ' ');
      env.detected = true; return;
    }
    const localInfo = await api.detectLocal();
    if (localInfo) { env.mode = 'local'; env.detected = true; setBadge('Local Python connected'); return; }
    env.mode = 'static'; env.detected = true; setBadge('Browser only');
  }
  async function loadRecorded() {
    // A derived copy of recorded-example.json in a script file, so the example also opens from a file: URL.
    if (window.FYD_RECORDED_EXAMPLE && window.FYD_RECORDED_EXAMPLE.outcomes) { env.recorded = window.FYD_RECORDED_EXAMPLE; return; }
    if (location.protocol === 'file:') return;
    try { const data = await api.staticJson('recorded-example.json'); if (data && data.outcomes) env.recorded = data; } catch (_) { env.recorded = null; }
  }
  async function boot() {
    $('panel-research').innerHTML = V.renderResearch(window.ERF_STUDY || null);
    $('panel-how').innerHTML = V.renderHow();
    const notes = V.renderPredictionNotes();
    $('prediction-privacy').textContent = notes.privacy; $('prediction-cannot').textContent = notes.cannot;
    renderOverview();
    show(routeFromHash() || 'overview', { focus: false });
    await Promise.all([detect(), loadRecorded()]);
    renderOverview();
    show(routeFromHash() || 'overview', { focus: false });
    const classic = document.querySelector('[data-classic="1"]'); if (classic && !env.hosted) classic.removeAttribute('href');
  }
  window.FYD_BOOTED = true;
  boot();
})();
