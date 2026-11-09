/* Sherwan Ali. The state machine of the hosted "Analyze a dataset" flow:
   Data -> Target and split -> Models and policy -> Run -> Results and export.
   It takes an API client, so tests drive it with a fake. Nothing is uploaded or trained unless an action is called
   for it, and no control change calls the network. The access tokens live in closure variables only. */
(function (root) {
  'use strict';
  const U = root.FYD.util, C = root.FYD.content, MODELS = root.FYD.MODELS;
  const STEPS = ['data', 'target', 'models', 'run', 'results'];
  const LIVE = new Set(['queued', 'running']);
  const RESULT_STATES = new Set(['succeeded', 'partial']);
  const POLL_MS = 1500;

  /* ---- pure helpers ---- */
  function parsePercents(text) {
    const raw = String(text == null ? '' : text).trim();
    if (!raw) return { values: [], error: null };
    const parts = raw.split(/[\s,;]+/).filter(Boolean);
    if (parts.length > 10) return { values: null, error: 'Use at most 10 review budgets.' };
    const values = [];
    for (const p of parts) {
      if (!/^\d+(\.\d+)?$/.test(p)) return { values: null, error: 'Review budgets are percentages such as 5, 10, 20.' };
      const n = Number(p);
      if (!(n > 0 && n <= 100)) return { values: null, error: 'Each review budget must be above 0 and at most 100 percent.' };
      values.push(n / 100);
    }
    if (new Set(values).size !== values.length) return { values: null, error: 'List each review budget once.' };
    return { values, error: null };
  }
  function parseCosts(costs) {
    const fp = String(costs.fp == null ? '' : costs.fp).trim(), fn = String(costs.fn == null ? '' : costs.fn).trim(), rv = String(costs.review == null ? '' : costs.review).trim();
    if (!fp && !fn && !rv) return { value: null, errors: [] };
    const errors = [];
    const one = (text, field, label) => { if (text === '') return null; const n = Number(text); if (!/^\d*\.?\d+(e[+-]?\d+)?$/i.test(text) || !Number.isFinite(n) || n < 0) { errors.push({ field, message: `${label} must be a number of zero or more.` }); return null; } return n; };
    const f = one(fp, 'analyze-cost-fp', 'Cost per false alarm'), g = one(fn, 'analyze-cost-fn', 'Cost per missed positive'), r = one(rv, 'analyze-cost-review', 'Cost per flagged case');
    if ((fp === '') !== (fn === '')) errors.push({ field: fp === '' ? 'analyze-cost-fp' : 'analyze-cost-fn', message: 'Give both error costs, or leave both blank.' });
    if (!errors.length && fp === '' && fn === '' && rv !== '') errors.push({ field: 'analyze-cost-fp', message: 'A review cost needs both error costs. Give all three, or leave all blank.' });
    if (errors.length) return { value: null, errors };
    const value = { false_positive_cost: f, false_negative_cost: g };
    if (r != null) value.review_cost = r;
    return { value, errors: [] };
  }
  /* Class counts for a column of a comma-delimited text file, only when its header matches the service profile. */
  function countLabelsFromCsv(text, column, schemaNames) {
    try {
      const M = root.RareEventMetrics;
      if (!M || typeof text !== 'string') return null;
      const parsed = M.parseCSV(text);
      // The service reader drops a first column that holds only row numbers, so the file may have one extra leading header.
      // That column is never offered by the page: the choices come from the service schema, which no longer lists it.
      const skip = schemaNames && parsed.headers.length === schemaNames.length + 1 ? 1 : 0;
      if (schemaNames && (parsed.headers.length !== schemaNames.length + skip || schemaNames.some((h, i) => h !== parsed.headers[i + skip]))) return null;
      const idx = parsed.headers.indexOf(column, skip);
      if (idx < 0) return null;
      const counts = new Map();
      let missing = 0;
      parsed.rows.forEach(r => { const v = r[idx]; if (v === '' || v == null) { missing++; return; } counts.set(v, (counts.get(v) || 0) + 1); });
      const labels = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, 50).map(([label, count]) => ({ label, count }));
      return { column, labels, distinct: counts.size, missing, source: 'browser' };
    } catch (_) { return null; }
  }
  function initialState() {
    return {
      mode: 'hosted', step: 'data', ui: { confirmDelete: false },
      version: null,
      samples: { phase: 'idle', items: [], error: null },
      data: { phase: 'empty', source: null, datasetId: null, profile: null, limits: null, expiresAt: null, error: null },
      classes: null,
      target: { column: '', positive: '', negative: '', oneVsRest: false, roles: {}, groupColumn: '', timeColumn: '', splitMode: 'stratified', seed: '42', missingPolicy: 'error', confirmed: false },
      models: { selected: ['logistic', 'rf_weighted'], calibration: 'none', reviewPercents: '10, 20', costs: { fp: '', fn: '', review: '' } },
      run: { phase: 'idle', jobId: null, state: null, stage: null, queuePosition: null, elapsed: null, completed: [], current: null, selected: [], limits: null, reason: null, notice: null, contactLost: false, checkedCount: 0 },
      result: null,
      errors: [], errorsVersion: 0, status: '', statusVersion: 0, focus: { id: null, n: 0 }, busy: false
    };
  }

  function validateTarget(state) {
    const t = state.target, e = [];
    const cols = state.data.profile ? (state.data.profile.schema_columns || []).map(c => c.column) : [];
    if (!t.column) e.push({ field: 'target-column', message: 'Choose the target column.' });
    else if (cols.length && !cols.includes(t.column)) e.push({ field: 'target-column', message: 'The target column is not in the file.' });
    if (!t.positive) e.push({ field: 'positive-label', message: 'Choose the positive class: the label of the cases you want to find.' });
    if (!t.oneVsRest) {
      if (!t.negative) e.push({ field: 'negative-label', message: 'Choose the negative class, or choose one against the rest.' });
      else if (t.negative === t.positive) e.push({ field: 'negative-label', message: 'The negative class must differ from the positive class.' });
    }
    const mode = t.splitMode;
    if (!C.SPLIT_MODES.some(m => m.id === mode)) e.push({ field: 'split-mode', message: 'Choose a split question.' });
    if (mode === 'group' && !t.groupColumn) e.push({ field: 'group-column', message: 'A group split needs a group column.' });
    if (mode === 'chronological' && !t.timeColumn) e.push({ field: 'time-column', message: 'A chronological split needs a time column.' });
    if (mode === 'chronological' && t.groupColumn) e.push({ field: 'group-column', message: 'A group column cannot be combined with a chronological split. Clear it or choose another split.' });
    [['groupColumn', 'group-column', 'group'], ['timeColumn', 'time-column', 'time']].forEach(([key, field, label]) => {
      if (t[key] && t[key] === t.column) e.push({ field, message: `The ${label} column cannot be the target.` });
    });
    if (t.groupColumn && t.groupColumn === t.timeColumn) e.push({ field: 'time-column', message: 'The group and time columns must differ.' });
    if (!/^-?\d+$/.test(String(t.seed).trim())) e.push({ field: 'seed', message: 'The seed must be a whole number.' });
    if (!t.confirmed) e.push({ field: 'confirm-target', message: 'Confirm the target, the positive class and the columns you leave out.' });
    return e;
  }
  function validateModels(state) {
    const m = state.models, e = [], limit = state.data.limits ? state.data.limits.max_models_per_job : null;
    if (!m.selected.length) e.push({ field: 'model-list', message: 'Choose at least one model.' });
    if (limit && m.selected.length > limit) e.push({ field: 'model-list', message: `Choose at most ${limit} models for one run.` });
    const known = new Set(MODELS.map(x => x.id));
    if (m.selected.some(id => !known.has(id))) e.push({ field: 'model-list', message: 'A selected model is not known to this version.' });
    if (m.calibration === 'sigmoid' && state.target.splitMode === 'chronological') e.push({ field: 'calibration', message: 'Sigmoid calibration is not available with a chronological split. Choose none, or another split.' });
    const p = parsePercents(m.reviewPercents);
    if (p.error) e.push({ field: 'review-budgets', message: p.error });
    parseCosts(m.costs).errors.forEach(x => e.push(x));
    return e;
  }
  function buildJobRequest(state) {
    const t = state.target, m = state.models;
    const body = { dataset_id: state.data.datasetId, target: t.column, positive_label: t.positive, models: [...m.selected], seed: parseInt(t.seed, 10), split_mode: t.splitMode, calibration: m.calibration };
    if (t.oneVsRest) body.one_vs_rest = true; else body.negative_label = t.negative;
    const ex = Object.keys(t.roles).filter(c => t.roles[c] === 'exclude' && c !== t.column), ids = Object.keys(t.roles).filter(c => t.roles[c] === 'id' && c !== t.column);
    if (ex.length) body.exclude_columns = ex;
    if (ids.length) body.id_columns = ids;
    if (t.groupColumn) body.group_column = t.groupColumn;
    if (t.timeColumn) body.time_column = t.timeColumn;
    if (t.missingPolicy && t.missingPolicy !== 'error') body.missing_policy = t.missingPolicy;
    const fr = parsePercents(m.reviewPercents).values;
    if (fr && fr.length) body.review_fractions = fr;
    const costs = parseCosts(m.costs).value;
    if (costs) body.costs = costs;
    return body;
  }
  /* A readable sentence for a failed call. HTTP texts come from fixed wording; the service message is added, escaped later. */
  function describeError(err, context) {
    if (!err) return 'Something went wrong.';
    if (err.kind === 'network') return err.message;
    if (err.kind === 'timeout') return 'The service did not answer in time. Try again in a moment.';
    const base = C.HTTP_TEXT[err.status] || 'The service could not complete the request.';
    const retry = err.retryAfter != null ? ` Try again in about ${U.duration(err.retryAfter)}.` : '';
    const lead = err.status === 429 && err.code === 'DATASET_BUSY' ? 'This dataset already has an analysis that has not finished.'
      : err.status === 429 && err.code === 'QUEUE_FULL' ? 'The service is busy with another analysis and its waiting place is taken.'
      : err.status === 429 && err.code === 'DATASET_LIMIT_REACHED' ? 'The service is holding the most datasets it allows.'
      : err.status === 503 && err.code === 'STORAGE_FULL' ? 'The service has no room for another file right now.'
      : err.status === 410 ? 'That item was removed from the service after its lifetime.'
      : err.status === 413 ? (err.details && err.details.limit_bytes ? `The file is larger than the limit of ${U.megabytes(err.details.limit_bytes)}.` : base)
      : base;
    const plainCode = err.code && C.REASONS[err.code] ? ' ' + C.REASONS[err.code] : '';
    const msg = err.message && err.message !== lead ? ` Service message: ${err.message}` : '';
    return `${lead}${plainCode}${retry}${msg}`.trim();
  }

  /* ---- the flow ---- */
  function createAnalysisFlow(deps) {
    const api = deps.api;
    const schedule = deps.schedule || ((fn, ms) => setTimeout(fn, ms));
    const unschedule = deps.unschedule || (h => clearTimeout(h));
    const readText = deps.readText || (async bytes => new TextDecoder('utf-8', { fatal: false }).decode(bytes));
    let state = initialState();
    const secrets = { dataset: null, job: null };      // never exposed, never serialized
    let pending = null;                                // { kind, name, size, file?, text?, bytes? }
    let pollHandle = null, pollToken = 0, textForCounts = null, listeners = new Set();
    function emit() { listeners.forEach(fn => fn(state)); }
    function setErrors(errors) { state.errors = errors; state.errorsVersion += errors.length ? 1 : 0; }
    function say(text) { state.status = U.markBidi(text); state.statusVersion++; }
    /* Ask the page to move focus to one element once, after the next draw. Only the actions below call it, so a later control change never moves focus. */
    function focusOn(id) { state.focus = { id, n: state.focus.n + 1 }; }
    function stopPolling() { pollToken++; if (pollHandle != null) { unschedule(pollHandle); pollHandle = null; } }

    function resetDownstream() {
      state.target = initialState().target; state.classes = null; textForCounts = null;
      state.models.selected = initialState().models.selected; state.result = null;
      state.run = initialState().run;
    }
    function computeClasses(column) {
      const p = state.data.profile;
      if (!p) return null;
      if (p.classes && p.classes.column === column) return { column, labels: p.classes.labels, distinct: p.classes.distinct_labels, missing: p.classes.missing_target_rows, source: 'service', truncated: !!p.classes.labels_truncated };
      return textForCounts ? countLabelsFromCsv(textForCounts, column, (p.schema_columns || []).map(c => c.column)) : null;
    }
    function applyClassDefaults(column) {
      const cls = computeClasses(column);
      state.classes = cls || { column, labels: null, distinct: null, missing: null, source: 'unknown' };
      const t = state.target;
      t.positive = ''; t.negative = ''; t.oneVsRest = false;
      if (cls && cls.labels && cls.labels.length >= 2) {
        const p = state.data.profile;
        const suggested = p && p.classes && p.classes.column === column && p.classes.suggested_positive ? p.classes.suggested_positive.label : null;
        // The rarest label is only a suggestion; the person confirms it. For other columns, the rarest counted label.
        const rarest = suggested != null ? suggested : [...cls.labels].sort((a, b) => a.count - b.count || (a.label < b.label ? -1 : 1))[0].label;
        t.positive = rarest;
        if (cls.distinct === 2) t.negative = cls.labels.find(l => l.label !== rarest).label; else t.oneVsRest = true;
      }
      t.confirmed = false;
    }
    function applyProfile(profile) {
      const t = state.target;
      const suggestion = profile.target && profile.target.suggestion && profile.target.suggestion.column;
      t.column = suggestion || (profile.target && profile.target.column) || '';
      t.roles = {};
      (profile.suggested_exclusions || []).forEach(x => { t.roles[x.column] = 'exclude'; });
      applyClassDefaults(t.column);
    }

    const flow = {
      getState: () => state,
      subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
      setVersion(info) { state.version = info; emit(); },
      /* A step is complete when what it asks for is present and valid. Going forward needs every earlier step complete. */
      stepComplete(step) {
        if (step === 'data') return state.data.phase === 'ready';
        if (step === 'target') return validateTarget(state).length === 0;
        if (step === 'models') return validateModels(state).length === 0;
        if (step === 'run') return !!state.result;
        return false;
      },
      canReach(step) {
        const to = STEPS.indexOf(step);
        if (to < 0) return false;
        for (let i = 0; i < to; i++) if (!flow.stepComplete(STEPS[i])) return false;
        return true;
      },
      setConfirmDelete(on) { state.ui.confirmDelete = !!on; emit(); },
      /* retry: the person asked again, so the result is announced and focus goes to the first sample or back to the button. */
      async loadSamples(retry) {
        if (state.samples.phase === 'loading' || (state.samples.phase === 'ready' && !retry)) return;
        state.samples = { phase: 'loading', items: [], error: null }; emit();
        try {
          const d = await api.samples();
          const byName = new Map((d.samples || []).map(s => [s.name, s]));
          const ordered = [];
          (d.order || []).forEach(cat => (d.samples || []).filter(s => s.category === cat).forEach(s => ordered.push(s)));
          (d.samples || []).forEach(s => { if (!ordered.includes(s)) ordered.push(s); });
          state.samples = { phase: 'ready', items: ordered.filter(s => byName.has(s.name)), error: null };
        } catch (e) { state.samples = { phase: 'error', items: [], error: describeError(e) }; }
        if (retry) {
          const s = state.samples;
          if (s.phase === 'ready') { say(`The sample list loaded: ${U.plural(s.items.length, 'sample')}.`); if (s.items.length) focusOn('sample-' + s.items[0].name); }
          else { say('The sample list could not be loaded. Try again.'); focusOn('retry-samples'); }
        }
        emit();
      },
      retrySamples() { return flow.loadSamples(true); },
      selectFile(file) {
        if (!file) return;
        pending = { kind: 'file', name: String(file.name || 'upload'), size: file.size, file };
        state.data.source = { kind: 'file', name: pending.name, size: file.size };
        state.data.importSelection={};state.data.importChoices=null;
        state.data.error = null;
        if (state.data.phase !== 'ready' && state.data.phase !== 'uploading') state.data.phase = 'selected';
        state.data.selectedNotUploaded = true;
        setErrors([]); say(`Selected ${pending.name}. It has not been uploaded. Choose Upload and profile when you are ready.`); emit();
      },
      selectSample(name) {
        const item = state.samples.items.find(s => s.name === name);
        if (!item) return;
        pending = { kind: 'sample', name: item.name, size: null, label: item.label };
        state.data.source = { kind: 'sample', name: item.name, label: item.label, size: null };
        state.data.importSelection={};state.data.importChoices=null;
        state.data.error = null;
        if (state.data.phase !== 'ready' && state.data.phase !== 'uploading') state.data.phase = 'selected';
        state.data.selectedNotUploaded = true;
        setErrors([]); say(`Selected the bundled sample ${item.label}. It has not been uploaded. Choose Upload and profile when you are ready.`); emit();
      },
      setImportSelection(kind,value) {
        if(!['sheet','member'].includes(kind))return;
        state.data.importSelection={...(state.data.importSelection||{}),[kind]:value};
        setErrors([]);emit();
      },
      async uploadSelected() {
        if (!pending) { setErrors([{ field: 'file-input', message: 'Choose a file or a bundled sample first.' }]); emit(); return false; }
        if (state.busy) return false;
        state.busy = true; state.data.phase = 'uploading'; setErrors([]); say('Uploading and profiling the file. This can take a few seconds.'); emit();
        const previous = secrets.dataset && state.data.datasetId ? { id: state.data.datasetId, cap: secrets.dataset } : null;
        try {
          let bytes, name = pending.name, text = null;
          if (pending.kind === 'sample') { text = await api.sampleText(pending.name); bytes = new TextEncoder().encode(text); }
          else { bytes = await pending.file.arrayBuffer(); if (/\.(csv|tsv|txt|data|dat)$/i.test(name) && bytes.byteLength <= 8 * 1048576) text = await readText(bytes); }
          const created = await api.createDataset(bytes, name, state.data.importSelection);
          if (previous) { try { await api.deleteDataset(previous.id, previous.cap); } catch (_) { /* the old dataset expires on its own */ } }
          secrets.dataset = created.capability; stopPolling(); secrets.job = null;
          state.data = { phase: 'ready', source: state.data.source, datasetId: created.dataset_id, profile: created.profile, limits: created.limits, expiresAt: created.expires_at, error: null, selectedNotUploaded: false };
          resetDownstream(); textForCounts = text; applyProfile(created.profile);
          say(`Profile ready: ${U.int(created.profile.rows)} rows, ${U.int(created.profile.columns)} columns. Review the parsing notes, then continue.`);
          focusOn('next-data');
        } catch (e) {
          if(e.code==='APP_IMPORT_SELECTION_REQUIRED' && e.details && Array.isArray(e.details.choices)) {
            state.data.importChoices={kind:e.details.selection_kind,choices:e.details.choices};
          }
          state.data.phase = previous ? 'ready' : 'error'; state.data.error = describeError(e, 'upload');
          setErrors([{ field: 'file-input', message: state.data.error }]);
          if (!previous) state.data.datasetId = null;
          say('The upload did not work.');
        }
        state.busy = false; emit(); return state.data.phase === 'ready';
      },
      async discardDataset() {
        if (!secrets.dataset || !state.data.datasetId) return false;
        const id = state.data.datasetId, cap = secrets.dataset;
        stopPolling();
        if (state.run.phase === 'live' || state.run.phase === 'starting') { try { await api.cancelJob(state.run.jobId, secrets.job); } catch (_) { /* deleting the dataset cancels it as well */ } }
        let message;
        try { await api.deleteDataset(id, cap); message = 'Your uploaded data was deleted from the service.'; }
        catch (e) { message = e && e.status === 404 ? 'The service no longer holds this dataset.' : 'The service could not confirm the deletion. The upload is removed after its lifetime.'; }
        secrets.dataset = null; secrets.job = null; pending = null; textForCounts = null;
        const samples = state.samples, version = state.version, statusVersion = state.statusVersion, errorsVersion = state.errorsVersion, focusN = state.focus.n;
        state = initialState(); state.samples = samples; state.version = version; state.statusVersion = statusVersion; state.errorsVersion = errorsVersion; state.focus.n = focusN;
        say(message); emit(); return true;
      },
      setTargetColumn(column) {
        state.target.column = column; state.target.confirmed = false; delete state.target.roles[column];
        if (state.target.groupColumn === column) state.target.groupColumn = '';
        if (state.target.timeColumn === column) state.target.timeColumn = '';
        applyClassDefaults(column); setErrors([]); emit();
      },
      /* silent: a text box updates state while you type without redrawing the page under your cursor. */
      setField(group, key, value, silent) {
        if (!state[group] || !(key in state[group])) throw Error('Unknown field.');
        state[group][key] = value;
        if (group === 'target' && key !== 'confirmed') state.target.confirmed = false;
        if (key === 'oneVsRest' && value) state.target.negative = '';
        if (!silent) emit();
      },
      /* A split question that does not use a group or time column clears it, so no stale column is sent. */
      setSplitMode(mode) {
        const t = state.target;
        t.splitMode = mode; t.confirmed = false;
        if (mode !== 'group') t.groupColumn = '';
        if (mode !== 'chronological') t.timeColumn = '';
        setErrors([]); emit();
      },
      setPositive(label) {
        state.target.positive = label; state.target.confirmed = false;
        const cls = state.classes;
        if (cls && cls.labels && cls.distinct === 2) { const other = cls.labels.find(l => l.label !== label); state.target.negative = other ? other.label : ''; state.target.oneVsRest = false; }
        emit();
      },
      setColumnRole(column, role) {
        if (role === 'include') delete state.target.roles[column]; else state.target.roles[column] = role;
        state.target.confirmed = false; emit();
      },
      setCost(key, value) { if (!(key in state.models.costs)) throw Error('Unknown cost.'); state.models.costs[key] = value; },
      toggleModel(id, on) {
        const set = new Set(state.models.selected);
        if (on) set.add(id); else set.delete(id);
        state.models.selected = MODELS.map(m => m.id).filter(x => set.has(x)); emit();
      },
      validateStep(step) { return step === 'target' ? validateTarget(state) : step === 'models' ? validateModels(state) : []; },
      /* Moves one step forward if the step you are leaving is valid. Never starts a run. */
      next() {
        const i = STEPS.indexOf(state.step);
        let errors = [];
        if (state.step === 'data') { if (state.data.phase !== 'ready') errors = [{ field: 'file-input', message: 'Upload and profile a dataset before continuing.' }]; }
        else if (state.step === 'target') errors = validateTarget(state);
        else if (state.step === 'models') errors = validateModels(state);
        else if (state.step === 'run') { if (!state.result) errors = [{ field: 'start-run', message: 'Start the analysis and wait for it to finish to see results.' }]; }
        setErrors(errors);
        if (errors.length) { say(`${errors.length} ${errors.length === 1 ? 'problem needs' : 'problems need'} attention.`); emit(); return false; }
        const to = STEPS[Math.min(STEPS.length - 1, i + 1)];
        state.step = to; say(`Step ${STEPS.indexOf(to) + 1} of 5.`); emit(); return true;
      },
      back() {
        const i = STEPS.indexOf(state.step);
        if (i === 0) return false;
        setErrors([]); state.step = STEPS[i - 1]; say(`Step ${i} of 5. Your choices were kept.`); emit(); return true;
      },
      goTo(step) {
        const to = STEPS.indexOf(step), cur = STEPS.indexOf(state.step);
        if (to < 0 || (to > cur && !flow.canReach(step))) return false;
        setErrors([]); state.step = step; say(`Step ${to + 1} of 5.`); emit(); return true;
      },
      async startRun() {
        if (state.busy || state.run.phase === 'starting' || state.run.phase === 'live') return false;
        const errors = [...validateTarget(state), ...validateModels(state)];
        if (state.data.phase !== 'ready') errors.unshift({ field: 'file-input', message: 'Upload and profile a dataset first.' });
        setErrors(errors);
        if (errors.length) { say(`${errors.length} ${errors.length === 1 ? 'problem needs' : 'problems need'} attention.`); emit(); return false; }
        const body = buildJobRequest(state);
        state.run = Object.assign(initialState().run, { phase: 'starting', selected: [...body.models], limits: state.data.limits ? { model: state.data.limits.per_model_timeout_seconds, job: state.data.limits.job_timeout_seconds } : null });
        state.result = null; state.busy = true; say('Starting the analysis.'); emit();
        try {
          const accepted = await api.createJob(secrets.dataset, body);
          secrets.job = accepted.capability;
          state.run.jobId = accepted.job_id; state.run.state = accepted.state; state.run.phase = 'live';
          state.busy = false; say('The analysis was accepted and is queued.'); focusOn('run-title'); emit();
          flow.beginPolling(); return true;
        } catch (e) {
          state.busy = false; state.run = initialState().run;
          if (e.status === 410 && e.code === 'DATASET_EXPIRED') { state.data.phase = 'expired'; state.data.error = 'Your dataset was removed after its lifetime. Upload it again.'; state.step = 'data'; }
          else if (e.status === 404) { state.data.phase = 'expired'; state.data.error = 'The service does not hold this dataset any more, for example after a restart. Upload it again.'; state.step = 'data'; }
          state.run.notice = describeError(e, 'start');
          setErrors([{ field: e.status === 429 ? 'start-run' : 'start-run', message: state.run.notice }]);
          say('The analysis was not started.'); emit(); return false;
        }
      },
      beginPolling() {
        stopPolling();
        const token = pollToken;
        const tick = async () => {
          if (token !== pollToken) return;
          let delay = POLL_MS;
          try {
            const s = await api.jobStatus(state.run.jobId, secrets.job);
            if (token !== pollToken) return;
            flow.applyStatus(s); state.run.contactLost = false; state.run.checkedCount++;
            if (!LIVE.has(s.state)) { emit(); await flow.finishRun(s); return; }
            emit();
          } catch (e) {
            if (token !== pollToken) return;
            if (e.status === 404) { stopPolling(); state.run.phase = 'ended'; state.run.state = 'lost'; focusOn('ended-title'); state.run.reason = { code: 'SERVICE_FORGOT', message: 'The service no longer knows this analysis. It may have been restarted, or the access token is gone after a reload.' }; say('The analysis cannot be found.'); emit(); return; }
            if (e.status === 410) { stopPolling(); state.run.phase = 'ended'; state.run.state = 'expired'; state.run.reason = null; focusOn('ended-title'); say('The result expired.'); emit(); return; }
            state.run.checkedCount++;
            if (e.retryAfter != null) delay = Math.max(delay, e.retryAfter * 1000);
            else if (e.kind === 'network' || e.kind === 'timeout') delay = Math.min(15000, POLL_MS * 2 ** Math.min(4, (state.run.failedChecks || 0) + 1));
            state.run.failedChecks = (state.run.failedChecks || 0) + 1;
            if (state.run.failedChecks >= 3) { state.run.contactLost = true; state.run.notice = 'The page lost contact with the service. The analysis may still be running; the page keeps trying.'; say('Lost contact with the service. Still trying.'); }
            emit();
          }
          if (token === pollToken) pollHandle = schedule(tick, delay);
        };
        pollHandle = schedule(tick, 600);
      },
      applyStatus(s) {
        const r = state.run, p = s.progress || {};
        r.state = s.state; r.queuePosition = s.queue_position == null ? null : s.queue_position;
        r.elapsed = typeof s.elapsed_seconds === 'number' ? s.elapsed_seconds : r.elapsed;
        r.stage = p.stage || r.stage; r.current = p.current_model || null;
        r.completed = Array.isArray(p.models_completed) ? p.models_completed.map(m => ({ model: m.model, status: m.status })) : (Array.isArray(s.models) ? s.models : r.completed);
        if (Array.isArray(s.models_selected) && s.models_selected.length) r.selected = s.models_selected;
        r.reason = s.reason || null; r.warnings = s.warnings || [];
        r.failedChecks = 0;
        const label = LIVE.has(s.state) ? (s.state === 'queued' ? `Queued${r.queuePosition != null ? ', position ' + r.queuePosition : ''}.` : `Running: ${C.STAGE_TEXT[r.stage] || 'working'}${r.current ? ' (' + r.current + ')' : ''}. ${r.completed.length} of ${r.selected.length || '?'} models finished.`) : C.JOB_STATE_TEXT[s.state] || s.state;
        say(label);
      },
      async finishRun(s) {
        stopPolling();
        state.run.phase = 'ended';
        if (RESULT_STATES.has(s.state)) {
          try {
            const r = await api.jobResult(state.run.jobId, secrets.job);
            if (!r.ready) { state.run.phase = 'live'; emit(); flow.beginPolling(); return; }
            state.result = { report: r.data, provenance: 'fresh_app_run', state: s.state };
            state.step = 'results'; say(s.state === 'partial' ? 'Finished with a partial result. Results are shown.' : 'Finished. Results are shown.');
          } catch (e) {
            if (e.status === 410) { state.run.state = 'expired'; say('The result expired before it could be read.'); }
            else { state.run.notice = 'The analysis finished but the report could not be read. ' + describeError(e); say('The report could not be read.'); }
          }
        } else {
          say(C.JOB_STATE_TEXT[s.state] || 'The analysis ended.');
        }
        if (state.step === 'run') focusOn('ended-title');
        emit();
      },
      async recheck() { if (state.run.jobId && secrets.job) { state.run.phase = 'live'; state.run.contactLost = false; flow.beginPolling(); emit(); } },
      async cancel() {
        const r = state.run;
        if (!(r.phase === 'live') || !r.jobId) return false;
        r.phase = 'cancelling'; say('Cancelling. The service stops the worker before it answers.'); emit();
        stopPolling();
        try {
          const s = await api.cancelJob(r.jobId, secrets.job);
          flow.applyStatus(s);
          state.run.phase = 'ended';
          if (!LIVE.has(s.state)) { await flow.finishRun(s); return true; }
          say('Cancellation requested.');
        } catch (e) {
          if (e.status === 404 || e.status === 410) { state.run.phase = 'ended'; state.run.state = e.status === 410 ? 'expired' : 'lost'; focusOn('ended-title'); say('The analysis is no longer on the service.'); }
          else { state.run.phase = 'live'; state.run.notice = 'The cancel request did not reach the service. ' + describeError(e); say('Cancel did not work. The analysis may still be running.'); flow.beginPolling(); }
        }
        emit(); return true;
      },
      backToSettings() {
        stopPolling(); state.run = initialState().run; state.result = null; secrets.job = null; setErrors([]);
        state.step = 'models'; say('Settings are kept. Change them and run again.'); emit();
      },
      /* Downloads need the job token, so they run inside the flow. */
      async download(kind) {
        if (!state.run.jobId || !secrets.job) throw Error('No run to download from.');
        if (!U.EXPORT_FILES[kind]) throw Error('Unknown download kind.');
        const created = state.result && state.result.report && Object.values(state.result.report.outcomes || {}).map(o => o.report && o.report.created_utc).find(Boolean);
        const blob = await api.exportBlob(state.run.jobId, secrets.job, kind);
        const name = U.downloadName(kind, state.data.source && (state.data.source.label || state.data.source.name), created);
        return { blob, name };
      },
      hasJob() { return !!(state.run.jobId && secrets.job); }
    };
    return flow;
  }
  const api = { createAnalysisFlow, initialState, parsePercents, parseCosts, countLabelsFromCsv, validateTarget, validateModels, buildJobRequest, describeError, STEPS, POLL_MS };
  root.FYD = root.FYD || {};
  root.FYD.flow = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
