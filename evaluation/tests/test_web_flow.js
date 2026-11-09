/* Sherwan Ali. The hosted analysis flow as a state machine, driven by a fake API client and a manual scheduler:
   all job states, 429, cancel, expiry, back navigation that keeps choices, and no automatic upload or training. */
'use strict';
const assert = require('node:assert/strict');
const S = require('./web_support.js');
const { FLOW, V, C, API } = S.load();
const { test } = S;

const clone = x => JSON.parse(JSON.stringify(x));
const DS = S.fixture('web-hosted-dataset.json'), RESULT = S.fixture('web-hosted-result.json');
const DS_CAP = 'DSCAP-5f3a9c0d1e2b4a6c8d7e9f0a1b2c3d4e5f60718', JOB_CAP = 'JOBCAP-9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0';
const HEADERS = DS.profile.schema_columns.map(c => c.column);
const CSV = HEADERS.join(',') + '\n' + Array.from({ length: 12 }, (_, i) => HEADERS.map(h => h === 'target' ? (i % 4 === 0 ? '1' : '0') : h === 'Gender' ? (i % 3 === 0 ? '1' : '0') : String(10 + i)).join(',')).join('\n') + '\n';
const SAMPLES = { samples: [{ name: 'liver_disease.csv', category: 'Medical', label: 'Liver disease', desc: 'find the less common group' }, { name: 'car_quality.csv', category: 'Manufacturing', label: 'Car quality', desc: 'rare good cars' }], order: ['Medical', 'Manufacturing'] };
const err = (status, code, message, extra = {}) => new API.ApiError(Object.assign({ kind: 'http', status, code, message, details: {} }, extra));

function status(state, extra = {}) {
  return Object.assign({ job_id: 'job_x', state, queue_position: null, elapsed_seconds: 1.5, limits: { per_model_timeout_seconds: 60, job_timeout_seconds: 300, queue_wait_seconds: 120 },
    progress: { stage: state === 'queued' ? null : 'fitting', models_selected: ['logistic', 'rf_weighted'], models_completed: [], current_model: 'logistic', limits: {}, warnings: [], elapsed_seconds: 1.5 },
    models: [], models_selected: ['logistic', 'rf_weighted'], warnings: [], reason: null, expires_at: null }, extra);
}
function makeApi(script = {}) {
  const calls = [];
  const rec = (name, args) => calls.push({ name, args });
  const next = (key, fallback) => { const q = script[key]; if (Array.isArray(q) && q.length) { const v = q.length > 1 ? q.shift() : q[0]; if (v instanceof Error) throw v; return typeof v === 'function' ? v() : v; } if (q instanceof Error) throw q; return fallback; };
  return {
    calls, script,
    async samples() { rec('samples', []); return next('samples', clone(SAMPLES)); },
    async sampleText(n) { rec('sampleText', [n]); return next('sampleText', CSV); },
    async createDataset(bytes, name) { rec('createDataset', [bytes.byteLength, name]); const d = next('createDataset', null); return d || Object.assign(clone(DS), { capability: DS_CAP }); },
    async deleteDataset(id, cap) { rec('deleteDataset', [id, cap]); return next('deleteDataset', { deleted: true }); },
    async createJob(cap, body) { rec('createJob', [cap, clone(body)]); return next('createJob', { job_id: 'job_x', capability: JOB_CAP, state: 'queued', poll_url: '/api/v1/jobs/job_x' }); },
    async jobStatus(id, cap) { rec('jobStatus', [id, cap]); return next('jobStatus', status('running')); },
    async cancelJob(id, cap) { rec('cancelJob', [id, cap]); return next('cancelJob', status('cancelled', { reason: { code: 'CANCELLED', message: 'The job was cancelled.' } })); },
    async jobResult(id, cap) { rec('jobResult', [id, cap]); return next('jobResult', { ready: true, data: clone(RESULT) }); },
    async exportBlob(id, cap, kind) { rec('exportBlob', [id, cap, kind]); return next('exportBlob', { size: 10, kind }); }
  };
}
function makeScheduler() {
  const queue = []; let id = 0;
  return { schedule: (fn, ms) => { queue.push({ id: ++id, fn, ms }); return id; }, unschedule: h => { const i = queue.findIndex(x => x.id === h); if (i >= 0) queue.splice(i, 1); },
    async tick() { const t = queue.shift(); if (!t) return false; await t.fn(); return true; }, pending: () => queue.length, delays: () => queue.map(q => q.ms) };
}
function newFlow(script) {
  const api = makeApi(script), sch = makeScheduler();
  const flow = FLOW.createAnalysisFlow({ api, schedule: sch.schedule, unschedule: sch.unschedule, readText: async b => new TextDecoder().decode(b) });
  return { flow, api, sch };
}
const fileOf = (name = 'loans.csv') => ({ name, size: CSV.length, arrayBuffer: async () => new TextEncoder().encode(CSV).buffer });
async function uploaded(script) {
  const t = newFlow(script);
  await t.flow.loadSamples();
  t.flow.selectSample('liver_disease.csv');
  await t.flow.uploadSelected();
  return t;
}
async function readyToRun(script) {
  const t = await uploaded(script);
  t.flow.setField('target', 'confirmed', true);
  assert.ok(t.flow.next(), 'target -> models');   // data -> target happens below
  return t;
}
async function atRunStep(script) {
  const t = await uploaded(script);
  assert.ok(t.flow.next(), 'data -> target');
  t.flow.setField('target', 'confirmed', true);
  assert.ok(t.flow.next(), 'target -> models');
  assert.ok(t.flow.next(), 'models -> run');
  assert.equal(t.flow.getState().step, 'run');
  return t;
}
const names = api => api.calls.map(c => c.name);

test('choosing a file or sample, and changing any control, never uploads or trains', async () => {
  const t = newFlow();
  await t.flow.loadSamples();
  t.flow.selectFile(fileOf());
  t.flow.selectSample('liver_disease.csv');
  assert.deepEqual(names(t.api), ['samples']);
  assert.equal(t.flow.getState().data.phase, 'selected');
  await t.flow.uploadSelected();
  const before = t.api.calls.length;
  // every control after upload: no network call
  t.flow.setTargetColumn('Gender'); t.flow.setPositive('1'); t.flow.setField('target', 'seed', '9', true); t.flow.setSplitMode('group');
  t.flow.setField('target', 'groupColumn', 'Age'); t.flow.setColumnRole('Albumin', 'exclude'); t.flow.setField('target', 'confirmed', true);
  t.flow.toggleModel('erf', true); t.flow.toggleModel('erf', false); t.flow.setField('models', 'calibration', 'sigmoid'); t.flow.setField('models', 'reviewPercents', '5, 10', true);
  t.flow.setCost('fp', '1'); t.flow.setCost('fn', '9'); t.flow.next(); t.flow.back();
  assert.equal(t.api.calls.length, before, 'no call after the upload except the explicit ones');
  assert.equal(names(t.api).filter(n => n === 'createJob').length, 0, 'no job without startRun');
  while (await t.sch.tick()) { /* nothing scheduled */ }
  assert.equal(names(t.api).filter(n => n === 'createJob' || n === 'jobStatus').length, 0);
});

test('uploading a bundled sample fetches it and uploads it the same way as a file', async () => {
  const t = await uploaded();
  assert.deepEqual(names(t.api), ['samples', 'sampleText', 'createDataset']);
  assert.equal(t.api.calls[1].args[0], 'liver_disease.csv');
  assert.equal(t.api.calls[2].args[1], 'liver_disease.csv');
  const st = t.flow.getState();
  assert.equal(st.data.phase, 'ready');
  assert.equal(st.data.profile.rows, 583);
  assert.equal(st.target.column, 'target', 'the suggestion is applied as a starting point');
  assert.equal(st.target.positive, '1');
  assert.equal(st.target.negative, '0');
  assert.equal(st.target.confirmed, false, 'but it still needs confirmation');
  assert.equal(st.classes.source, 'service');
  const f = newFlow(); await f.flow.loadSamples(); f.flow.selectFile(fileOf('loans.csv')); await f.flow.uploadSelected();
  assert.deepEqual(names(f.api), ['samples', 'createDataset']);
  assert.equal(f.api.calls[1].args[1], 'loans.csv');
});

test('capabilities never appear in state, status text, requests URLs or serialized state', async () => {
  const t = await atRunStep();
  await t.flow.startRun();
  await t.sch.tick();
  const dump = JSON.stringify(t.flow.getState());
  assert.ok(!dump.includes(DS_CAP) && !dump.includes(JOB_CAP), 'not in state');
  assert.ok(!t.flow.getState().status.includes('CAP'));
  // tokens are only passed as the separate bearer argument of the client calls
  const withCap = t.api.calls.filter(c => c.args.includes(DS_CAP) || c.args.includes(JOB_CAP));
  assert.ok(withCap.length >= 2);
  withCap.forEach(c => { assert.ok(['createJob', 'jobStatus', 'deleteDataset'].includes(c.name)); });
  assert.ok(!JSON.stringify(t.api.calls.find(c => c.name === 'createJob').args[1]).includes('CAP'), 'the job body holds no token');
});

test('upload errors are readable: 413, 422, 429 and 503', async () => {
  const cases = [[err(413, 'UPLOAD_TOO_LARGE', 'too big', { details: { limit_bytes: 8388608 } }), /larger than the limit of 8\.0 MB/],
    [err(422, 'APP_INPUT_UNREADABLE', 'The file could not be read as a table (ValueError).'), /could not use this input.*could not be read as a table/],
    [err(429, 'DATASET_LIMIT_REACHED', 'Too many datasets.', { retryAfter: 30 }), /holding the most datasets it allows.*about 30 seconds/],
    [err(503, 'STORAGE_FULL', 'Storage is full.'), /no room for another file/],
    [new API.ApiError({ kind: 'network', message: 'The service could not be reached. Check your connection and try again.' }), /could not be reached/]];
  for (const [error, pattern] of cases) {
    const t = newFlow({ createDataset: error });
    await t.flow.loadSamples(); t.flow.selectSample('liver_disease.csv');
    assert.equal(await t.flow.uploadSelected(), false);
    const st = t.flow.getState();
    assert.equal(st.data.phase, 'error');
    assert.match(st.errors[0].message, pattern);
    assert.equal(st.errors[0].field, 'file-input');
    assert.ok(st.errorsVersion > 0, 'the error summary can take focus');
  }
});

test('uploading again deletes the previous dataset only after the new one exists', async () => {
  const t = await uploaded({ createDataset: [Object.assign(clone(DS), { capability: DS_CAP, dataset_id: 'ds_first' }), Object.assign(clone(DS), { capability: 'DSCAP-second', dataset_id: 'ds_second' })] });
  t.flow.selectFile(fileOf('second.csv'));
  assert.equal(t.flow.getState().data.phase, 'ready', 'the first dataset stays usable until the second is uploaded');
  await t.flow.uploadSelected();
  const order = names(t.api).filter(n => n === 'createDataset' || n === 'deleteDataset');
  assert.deepEqual(order, ['createDataset', 'createDataset', 'deleteDataset']);
  assert.deepEqual(t.api.calls.find(c => c.name === 'deleteDataset').args, ['ds_first', DS_CAP]);
  assert.equal(t.flow.getState().data.datasetId, 'ds_second');
});

test('a failed second upload keeps the first dataset', async () => {
  const t = await uploaded({ createDataset: [Object.assign(clone(DS), { capability: DS_CAP }), err(422, 'APP_INPUT_UNREADABLE', 'bad')] });
  t.flow.selectFile(fileOf('bad.csv'));
  assert.equal(await t.flow.uploadSelected(), true);
  assert.equal(t.flow.getState().data.phase, 'ready');
  assert.match(t.flow.getState().data.error, /could not use this input/);
});

test('step validation: each problem is named, with the field it belongs to', async () => {
  const t = await uploaded();
  assert.ok(t.flow.next());                      // data -> target
  assert.equal(t.flow.next(), false);            // not confirmed
  assert.deepEqual(t.flow.getState().errors.map(e => e.field), ['confirm-target']);
  t.flow.setPositive(''); t.flow.setSplitMode('group'); t.flow.setField('target', 'seed', 'abc', true);
  assert.equal(t.flow.next(), false);
  const fields = t.flow.getState().errors.map(e => e.field);
  assert.ok(['positive-label', 'group-column', 'seed', 'confirm-target'].every(f => fields.includes(f)), fields.join());
  t.flow.setSplitMode('chronological');
  t.flow.setField('target', 'groupColumn', 'Age'); // stale value cannot be combined with a time split
  assert.ok(t.flow.validateStep('target').some(e => e.field === 'group-column'));
  t.flow.setSplitMode('stratified');
  assert.equal(t.flow.getState().target.groupColumn, '', 'a split question that does not use a group column clears it');
  assert.equal(t.flow.getState().target.timeColumn, '');
});

test('model and policy validation covers limits, calibration, budgets and costs', async () => {
  const t = await uploaded();
  t.flow.next(); t.flow.setField('target', 'confirmed', true); t.flow.next();
  assert.equal(t.flow.getState().step, 'models');
  t.flow.toggleModel('logistic', false); t.flow.toggleModel('rf_weighted', false);
  assert.ok(t.flow.validateStep('models').some(e => e.field === 'model-list'));
  ['logistic', 'rf_raw', 'rf_weighted', 'balanced_rf', 'hist_gradient_boosting', 'erf'].forEach(id => t.flow.toggleModel(id, true));
  assert.match(t.flow.validateStep('models').find(e => e.field === 'model-list').message, /at most 5/);
  t.flow.toggleModel('erf', false);
  t.flow.setField('models', 'reviewPercents', '0, 10', true);
  assert.ok(t.flow.validateStep('models').some(e => e.field === 'review-budgets'));
  t.flow.setField('models', 'reviewPercents', '5, 10, 5', true);
  assert.ok(t.flow.validateStep('models').some(e => /once/.test(e.message)));
  t.flow.setField('models', 'reviewPercents', '5, 10', true);
  t.flow.setCost('fp', '1'); assert.ok(t.flow.validateStep('models').some(e => e.field === 'analyze-cost-fn'));
  t.flow.setCost('fn', '-3'); assert.ok(t.flow.validateStep('models').some(e => /zero or more/.test(e.message)));
  t.flow.setCost('fn', '9'); t.flow.setField('models', 'calibration', 'sigmoid');
  assert.deepEqual(t.flow.validateStep('models'), []);
  t.flow.setField('target', 'splitMode', 'chronological');
  assert.ok(t.flow.validateStep('models').some(e => e.field === 'calibration'), 'sigmoid with a chronological split is refused up front');
});

test('percent and cost parsing', () => {
  assert.deepEqual(FLOW.parsePercents('5, 10 20'), { values: [0.05, 0.1, 0.2], error: null });
  assert.deepEqual(FLOW.parsePercents('').values, []);
  assert.ok(FLOW.parsePercents('101').error);
  assert.ok(FLOW.parsePercents('abc').error);
  assert.ok(FLOW.parsePercents('1,2,3,4,5,6,7,8,9,10,11').error);
  assert.deepEqual(FLOW.parseCosts({ fp: '', fn: '', review: '' }), { value: null, errors: [] });
  assert.deepEqual(FLOW.parseCosts({ fp: '1', fn: '5', review: '' }).value, { false_positive_cost: 1, false_negative_cost: 5 });
  assert.deepEqual(FLOW.parseCosts({ fp: '1', fn: '5', review: '0.5' }).value, { false_positive_cost: 1, false_negative_cost: 5, review_cost: 0.5 });
  assert.ok(FLOW.parseCosts({ fp: '', fn: '', review: '2' }).errors.length);
});

test('going back keeps every choice; going forward past an invalid step is refused', async () => {
  const t = await uploaded();
  t.flow.next();
  t.flow.setColumnRole('Albumin', 'exclude'); t.flow.setColumnRole('Age', 'id');
  t.flow.setField('target', 'seed', '7', true); t.flow.setSplitMode('feature_group');
  t.flow.setField('target', 'confirmed', true);
  t.flow.next();
  t.flow.toggleModel('erf', true); t.flow.setField('models', 'reviewPercents', '5', true); t.flow.setCost('fp', '2'); t.flow.setCost('fn', '8');
  t.flow.next();
  assert.equal(t.flow.getState().step, 'run');
  const snapshot = clone({ target: t.flow.getState().target, models: t.flow.getState().models });
  t.flow.back(); t.flow.back(); t.flow.back();
  assert.equal(t.flow.getState().step, 'data');
  assert.deepEqual(clone({ target: t.flow.getState().target, models: t.flow.getState().models }), snapshot, 'nothing was reset by going back');
  assert.equal(t.flow.goTo('run'), true, 'forward again is allowed while every step is still valid');
  t.flow.goTo('target'); t.flow.setPositive('0'); // a change that needs a new confirmation
  assert.equal(t.flow.goTo('models'), false, 'an unconfirmed target blocks the way forward');
  assert.equal(t.flow.getState().step, 'target');
  assert.equal(t.flow.getState().target.roles.Albumin, 'exclude');
});

test('the job request carries exactly the confirmed choices', async () => {
  const t = await uploaded();
  t.flow.next();
  t.flow.setColumnRole('Albumin', 'exclude'); t.flow.setColumnRole('Age', 'id');
  t.flow.setField('target', 'seed', '7', true); t.flow.setField('target', 'missingPolicy', 'drop_reported');
  t.flow.setField('target', 'confirmed', true); t.flow.next();
  t.flow.toggleModel('erf', true); t.flow.setField('models', 'calibration', 'sigmoid'); t.flow.setField('models', 'reviewPercents', '5, 20', true); t.flow.setCost('fp', '2'); t.flow.setCost('fn', '8'); t.flow.setCost('review', '0.5');
  t.flow.next();
  assert.equal(await t.flow.startRun(), true);
  const [cap, body] = t.api.calls.find(c => c.name === 'createJob').args;
  assert.equal(cap, DS_CAP);
  assert.deepEqual(body, { dataset_id: DS.dataset_id, target: 'target', positive_label: '1', models: ['logistic', 'rf_weighted', 'erf'], seed: 7, split_mode: 'stratified', calibration: 'sigmoid',
    negative_label: '0', exclude_columns: ['Albumin'], id_columns: ['Age'], missing_policy: 'drop_reported', review_fractions: [0.05, 0.2], costs: { false_positive_cost: 2, false_negative_cost: 8, review_cost: 0.5 } });
});

test('a multiclass target is one against the rest and says so', async () => {
  const ds = clone(DS);
  ds.capability = DS_CAP; ds.profile.classes.labels = [{ label: 'a', count: 50 }, { label: 'b', count: 30 }, { label: 'c', count: 5 }]; ds.profile.classes.distinct_labels = 3; ds.profile.classes.suggested_positive = { label: 'c', basis: 'rarest' };
  const t = await uploaded({ createDataset: [ds] });
  const st = t.flow.getState();
  assert.equal(st.target.oneVsRest, true); assert.equal(st.target.positive, 'c'); assert.equal(st.target.negative, '');
  t.flow.next(); t.flow.setField('target', 'confirmed', true); t.flow.next(); t.flow.next();
  await t.flow.startRun();
  const body = t.api.calls.find(c => c.name === 'createJob').args[1];
  assert.equal(body.one_vs_rest, true); assert.ok(!('negative_label' in body));
});

test('changing the target to another column recomputes classes in the browser when the file allows it', async () => {
  const t = await uploaded();
  t.flow.setTargetColumn('Gender');
  const st = t.flow.getState();
  assert.equal(st.classes.source, 'browser');
  assert.equal(st.classes.labels.reduce((a, l) => a + l.count, 0), 12);
  assert.ok(st.target.positive && st.target.negative && st.target.positive !== st.target.negative);
  assert.equal(st.target.confirmed, false);
  const u = await uploaded({ sampleText: ['not,a,matching,table\n1,2,3,4\n'] });
  u.flow.setTargetColumn('Gender');
  assert.equal(u.flow.getState().classes.source, 'unknown', 'when counts cannot be trusted the page asks for the label text');
  assert.equal(u.flow.getState().target.positive, '');
});

test('a leading row-number column dropped by the service reader is never offered, and class counts still work', async () => {
  const withIndex = ['row,' + HEADERS.join(',')].concat(CSV.trim().split('\n').slice(1).map((l, i) => `${i + 1},${l}`)).join('\n') + '\n';
  const t = await uploaded({ sampleText: [withIndex] });
  t.flow.setTargetColumn('Gender');
  const st = t.flow.getState();
  assert.equal(st.classes.source, 'browser');
  assert.equal(st.classes.labels.reduce((a, l) => a + l.count, 0), 12);
  const html = V.renderTarget((t.flow.next(), t.flow.getState()));
  assert.ok(!/id="role-row"/.test(html) && !html.includes('>row<'), 'the dropped column is not a choice');
  assert.ok(FLOW.countLabelsFromCsv(withIndex, 'row', HEADERS) === null, 'counting a column the service did not read is refused');
});

test('progress: queued, running with stage and models, then the result; no timer unless a job is live', async () => {
  const script = { jobStatus: [status('queued', { queue_position: 1 }), status('running', { progress: { stage: 'fitting', models_selected: ['logistic', 'rf_weighted'], models_completed: [{ model: 'logistic', status: 'succeeded' }], current_model: 'rf_weighted', limits: {}, warnings: [], elapsed_seconds: 4 }, elapsed_seconds: 4 }), status('succeeded', { progress: { stage: 'done', models_selected: ['logistic', 'rf_weighted'], models_completed: [{ model: 'logistic', status: 'succeeded' }, { model: 'rf_weighted', status: 'succeeded' }], current_model: null, limits: {}, warnings: [], elapsed_seconds: 7 }, elapsed_seconds: 7 })] };
  const t = await atRunStep(script);
  assert.equal(t.sch.pending(), 0, 'nothing polls before a run starts');
  await t.flow.startRun();
  assert.equal(t.sch.pending(), 1);
  await t.sch.tick();
  let st = t.flow.getState();
  assert.equal(st.run.state, 'queued'); assert.equal(st.run.queuePosition, 1); assert.match(st.status, /Queued, position 1/);
  await t.sch.tick();
  st = t.flow.getState();
  assert.equal(st.run.stage, 'fitting'); assert.equal(st.run.current, 'rf_weighted'); assert.equal(st.run.completed.length, 1); assert.equal(st.run.elapsed, 4);
  assert.match(st.status, /Fitting models \(rf_weighted\)\. 1 of 2 models finished/);
  const live = V.renderRunLive(st);
  assert.ok(!live.includes('%'), 'never a percentage');
  assert.match(live, /Models finished<\/dt><dd>1 of 2/);
  assert.match(live, /not a percentage and not an estimate/);
  assert.match(live, /Logistic regression<\/strong>: finished with a result/);
  assert.match(live, /Random forest, class-weighted<\/strong>: fitting now/);
  await t.sch.tick();
  st = t.flow.getState();
  assert.equal(st.run.phase, 'ended'); assert.equal(st.step, 'results');
  assert.equal(st.result.provenance, 'fresh_app_run'); assert.equal(st.result.state, 'succeeded');
  assert.equal(t.sch.pending(), 0, 'polling stopped');
  assert.ok(t.sch.delays().length === 0);
});

test('the poll interval is at least one second', async () => {
  const t = await atRunStep();
  await t.flow.startRun();
  assert.ok(FLOW.POLL_MS >= 1000);
  await t.sch.tick();
  assert.ok(t.sch.delays().every(ms => ms >= 1000));
});

for (const [state, reason, expectText] of [
  ['failed', { code: 'LABEL_POSITIVE_ABSENT', message: 'The positive label is not in the column.' }, /The analysis did not produce a result/],
  ['cancelled', { code: 'CANCELLED', message: 'The job was cancelled.' }, /The analysis was cancelled/],
  ['timed_out', { code: 'JOB_TIMEOUT', message: 'Too long.' }, /reached a time limit/],
  ['interrupted', { code: 'SERVICE_STOPPING', message: 'Stopped.' }, /service was stopped before the analysis finished/]]) {
  test(`state ${state} is explained in plain language, offers a way forward and fetches no result`, async () => {
    const t = await atRunStep({ jobStatus: [status(state, { reason })] });
    await t.flow.startRun(); await t.sch.tick();
    const st = t.flow.getState();
    assert.equal(st.run.phase, 'ended'); assert.equal(st.run.state, state); assert.equal(st.result, null);
    assert.equal(names(t.api).includes('jobResult'), false);
    const html = V.renderRun(st);
    assert.match(html, expectText);
    assert.ok(html.includes(reason.code));
    assert.match(html, /data-action="settings"|data-action="goto-data"/);
    assert.equal(st.step, 'run');
    assert.equal(t.sch.pending(), 0);
  });
}

test('a partial run shows its results and says some models did not produce one', async () => {
  const t = await atRunStep({ jobStatus: [status('partial', { reason: { code: 'SOME_MODELS_FAILED', message: 'x' } })] });
  await t.flow.startRun(); await t.sch.tick();
  const st = t.flow.getState();
  assert.equal(st.step, 'results'); assert.equal(st.result.state, 'partial');
  assert.match(V.renderResults(st), /Partial result: at least one model did not produce a result/);
});

test('expiry: a result that expired, a job removed by the sweep and a dataset removed after idling', async () => {
  const a = await atRunStep({ jobStatus: [err(410, 'JOB_EXPIRED', 'expired')] });
  await a.flow.startRun(); await a.sch.tick();
  assert.equal(a.flow.getState().run.state, 'expired');
  assert.match(V.renderRun(a.flow.getState()), /removed from the service after its lifetime/);
  const b = await atRunStep({ jobStatus: [status('succeeded')], jobResult: [err(410, 'JOB_EXPIRED', 'gone')] });
  await b.flow.startRun(); await b.sch.tick();
  assert.equal(b.flow.getState().run.state, 'expired'); assert.equal(b.flow.getState().result, null);
  const c = await atRunStep({ createJob: [err(410, 'DATASET_EXPIRED', 'idle')] });
  assert.equal(await c.flow.startRun(), false);
  assert.equal(c.flow.getState().step, 'data'); assert.equal(c.flow.getState().data.phase, 'expired');
  assert.match(V.renderData(c.flow.getState()), /Your dataset was removed after its lifetime\. Upload it again\./);
  const d = await atRunStep({ jobStatus: [err(404, 'NOT_FOUND', 'Not found.')] });
  await d.flow.startRun(); await d.sch.tick();
  assert.equal(d.flow.getState().run.state, 'lost');
  assert.match(V.renderRun(d.flow.getState()), /no longer knows this analysis/);
});

test('429 busy is a clear retry message, not an error state, and starts no polling', async () => {
  for (const [code, retry, pattern] of [['QUEUE_FULL', 15, /busy with another analysis.*about 15 seconds/], ['DATASET_BUSY', 5, /already has an analysis that has not finished.*about 5 seconds/]]) {
    const t = await atRunStep({ createJob: [err(429, code, 'busy', { retryAfter: retry, details: { retry_after_seconds: retry } })] });
    assert.equal(await t.flow.startRun(), false);
    const st = t.flow.getState();
    assert.equal(st.run.phase, 'idle'); assert.equal(st.step, 'run'); assert.equal(st.data.phase, 'ready', 'the dataset is kept');
    assert.match(st.errors[0].message, pattern);
    assert.equal(st.errors[0].field, 'start-run');
    assert.equal(t.sch.pending(), 0);
    assert.match(V.renderAnalyze(st, t.flow), /id="start-run"/, 'the button to try again is there');
    // trying again works
    t.api.script.createJob = [{ job_id: 'job_x', capability: JOB_CAP, state: 'queued', poll_url: '/x' }];
    assert.equal(await t.flow.startRun(), true);
  }
});

test('cancel stops polling, calls the API with the job token and ends in cancelled', async () => {
  const t = await atRunStep();
  await t.flow.startRun(); await t.sch.tick();
  assert.equal(t.flow.getState().run.phase, 'live');
  const done = t.flow.cancel();
  assert.equal(t.flow.getState().run.phase, 'cancelling');
  assert.equal(t.sch.pending(), 0, 'no poll while cancelling');
  await done;
  assert.deepEqual(t.api.calls.find(c => c.name === 'cancelJob').args, ['job_x', JOB_CAP]);
  const st = t.flow.getState();
  assert.equal(st.run.phase, 'ended'); assert.equal(st.run.state, 'cancelled'); assert.equal(st.result, null);
  assert.equal(await t.flow.cancel(), false, 'cancel is not repeated once the run has ended');
  assert.match(V.renderRun(st), /The analysis was cancelled/);
});

test('a cancel that does not reach the service leaves the run live and polling again', async () => {
  const t = await atRunStep({ cancelJob: [new API.ApiError({ kind: 'network', message: 'The service could not be reached.' })] });
  await t.flow.startRun(); await t.sch.tick();
  await t.flow.cancel();
  const st = t.flow.getState();
  assert.equal(st.run.phase, 'live');
  assert.match(st.run.notice, /cancel request did not reach the service/);
  assert.equal(t.sch.pending(), 1, 'polling resumed');
});

test('losing contact is reported, retried with backoff, and recovers', async () => {
  const net = new API.ApiError({ kind: 'network', message: 'The service could not be reached.' });
  const t = await atRunStep({ jobStatus: [net, net, net, status('running'), status('succeeded')] });
  await t.flow.startRun();
  for (let i = 0; i < 3; i++) await t.sch.tick();
  let st = t.flow.getState();
  assert.equal(st.run.contactLost, true); assert.match(st.run.notice, /lost contact/);
  assert.ok(t.sch.delays()[0] > FLOW.POLL_MS, 'backs off');
  await t.sch.tick();
  assert.equal(t.flow.getState().run.contactLost, false);
  await t.sch.tick();
  assert.equal(t.flow.getState().step, 'results');
});

test('downloads use the job token inside the flow and meaningful file names', async () => {
  const t = await atRunStep({ jobStatus: [status('succeeded')] });
  await t.flow.startRun(); await t.sch.tick();
  for (const [kind, name] of [['report_html', 'fyd-liver-disease-2026-11-09-report.html'], ['report_json', 'fyd-liver-disease-2026-11-09-report.json'], ['config_json', 'fyd-liver-disease-2026-11-09-config.json'], ['repro_bundle', 'fyd-liver-disease-2026-11-09-repro-bundle.zip']]) {
    const out = await t.flow.download(kind);
    assert.equal(out.name, name);
    assert.deepEqual(t.api.calls.filter(c => c.name === 'exportBlob').pop().args, ['job_x', JOB_CAP, kind]);
  }
  await assert.rejects(() => t.flow.download('rows_csv'));
});

test('deleting the uploaded data cancels, deletes, and clears the page state', async () => {
  const t = await atRunStep();
  await t.flow.startRun(); await t.sch.tick();
  assert.equal(await t.flow.discardDataset(), true);
  assert.deepEqual(t.api.calls.filter(c => c.name === 'deleteDataset').map(c => c.args), [[DS.dataset_id, DS_CAP]]);
  assert.ok(names(t.api).includes('cancelJob'));
  const st = t.flow.getState();
  assert.equal(st.data.phase, 'empty'); assert.equal(st.step, 'data'); assert.equal(st.result, null);
  assert.equal(t.sch.pending(), 0);
  assert.equal(await t.flow.discardDataset(), false);
  assert.match(st.status, /deleted from the service/);
  assert.equal(st.samples.phase, 'ready', 'the sample list is kept');
});

test('the run step offers a start button and nothing else starts it', async () => {
  const t = await atRunStep();
  const html = V.renderAnalyze(t.flow.getState(), t.flow);
  assert.match(html, /id="start-run" data-action="start">Start analysis<\/button>/);
  assert.match(html, /Nothing starts until you press the button/);
  assert.equal(names(t.api).includes('createJob'), false);
});

test('each step button has a full accessible name and spaced visible text', async () => {
  const t = await uploaded();
  const html = V.renderStepper(t.flow.getState(), t.flow);
  assert.match(html, /data-step="data" aria-label="Step 1 of 5: Data, current" aria-current="step"/);
  assert.match(html, /data-step="target" aria-label="Step 2 of 5: Target and split, available"/);
  assert.match(html, /data-step="models" aria-label="Step 3 of 5: Models and policy, locked" disabled/);
  assert.match(html, /<\/span> <span class="step-title">Data<\/span> <span class="step-state">Current<\/span>/);
  t.flow.next();
  assert.match(V.renderStepper(t.flow.getState(), t.flow), /aria-label="Step 1 of 5: Data, done"/);
});

test('completion of the upload is announced in words and asks for one focus move to the Continue button', async () => {
  const t = newFlow();
  await t.flow.loadSamples(); t.flow.selectSample('liver_disease.csv');
  const n0 = t.flow.getState().focus.n;
  await t.flow.uploadSelected();
  let st = t.flow.getState();
  assert.equal(st.status, 'Profile ready: 583 rows, 11 columns. Review the parsing notes, then continue.');
  assert.equal(st.focus.id, 'next-data'); assert.equal(st.focus.n, n0 + 1);
  assert.match(V.renderData(st), /<button type="button" class="btn btn-primary" id="next-data" data-action="next">/);
  const n1 = st.focus.n;
  // later control changes never ask for a focus move
  t.flow.next(); t.flow.setTargetColumn('Gender'); t.flow.setPositive('1'); t.flow.setSplitMode('group'); t.flow.setColumnRole('Albumin', 'exclude'); t.flow.setField('target', 'confirmed', true);
  t.flow.next(); t.flow.toggleModel('erf', true); t.flow.setField('models', 'calibration', 'sigmoid'); t.flow.back();
  assert.equal(t.flow.getState().focus.n, n1);
});

test('starting a run asks for focus on the progress heading, and an ended run on its heading', async () => {
  const t = await atRunStep({ jobStatus: [status('failed', { reason: { code: 'LABEL_POSITIVE_ABSENT', message: 'x' } })] });
  const n = t.flow.getState().focus.n;
  await t.flow.startRun();
  assert.equal(t.flow.getState().focus.id, 'run-title'); assert.equal(t.flow.getState().focus.n, n + 1);
  assert.match(V.renderRun(t.flow.getState()), /<h3 id="run-title" tabindex="-1">Analysis in progress<\/h3>/);
  await t.sch.tick();
  assert.equal(t.flow.getState().focus.id, 'ended-title'); assert.equal(t.flow.getState().focus.n, n + 2);
  assert.match(V.renderRun(t.flow.getState()), /<h3 id="ended-title" tabindex="-1">/);
  const c = await atRunStep();
  await c.flow.startRun(); await c.sch.tick();
  const before = c.flow.getState().focus.n;
  await c.flow.cancel();
  assert.equal(c.flow.getState().focus.id, 'ended-title'); assert.ok(c.flow.getState().focus.n > before, 'cancel moves focus off the button that disappeared');
  // results: reached by a step change, whose heading takes focus; the announcement says so
  const r = await atRunStep({ jobStatus: [status('succeeded')] });
  await r.flow.startRun(); await r.sch.tick();
  assert.equal(r.flow.getState().step, 'results'); assert.equal(r.flow.getState().status, 'Finished. Results are shown.');
  assert.match(V.renderResults(r.flow.getState()), /<h2 id="step-heading" tabindex="-1">5\. Results and export<\/h2>/);
});

test('an unreachable service when starting is an error summary item, the run is not stuck and Start can be retried', async () => {
  const net = new API.ApiError({ kind: 'network', message: 'The service could not be reached. Check your connection and try again.' });
  const t = await atRunStep({ createJob: [net] });
  const v0 = t.flow.getState().errorsVersion;
  assert.equal(await t.flow.startRun(), false);
  let st = t.flow.getState();
  assert.equal(st.run.phase, 'idle'); assert.equal(st.busy, false); assert.equal(st.data.phase, 'ready'); assert.equal(st.step, 'run');
  assert.ok(st.errorsVersion > v0, 'the summary takes focus');
  assert.equal(st.errors[0].field, 'start-run'); assert.match(st.errors[0].message, /could not be reached/);
  assert.equal(st.status, 'The analysis was not started.');
  const html = V.renderAnalyze(st, t.flow);
  assert.match(html, /<div class="error-summary" id="error-summary" role="alert" tabindex="-1">/);
  assert.match(html, /<a href="#start-run" data-action="focus-field" data-field="start-run">The service could not be reached/);
  assert.match(html, /id="start-run" data-action="start">Start analysis/);
  assert.equal(t.sch.pending(), 0);
  t.api.script.createJob = [{ job_id: 'job_x', capability: JOB_CAP, state: 'queued', poll_url: '/x' }];
  assert.equal(await t.flow.startRun(), true);
  st = t.flow.getState();
  assert.equal(st.run.phase, 'live'); assert.deepEqual(st.errors, []);
  assert.equal(names(t.api).filter(n => n === 'createJob').length, 2);
  // a timeout and a 503 behave the same way
  for (const e of [new API.ApiError({ kind: 'timeout', message: 'x' }), err(503, 'SERVICE_UNAVAILABLE', 'down', { retryAfter: 5 })]) {
    const u = await atRunStep({ createJob: [e] });
    assert.equal(await u.flow.startRun(), false); assert.equal(u.flow.getState().errors[0].field, 'start-run'); assert.equal(u.flow.getState().busy, false);
  }
});

test('the stepper marks the current step and locks steps that cannot be reached', async () => {
  const t = await uploaded();
  const html = V.renderStepper(t.flow.getState(), t.flow);
  assert.equal((html.match(/aria-current="step"/g) || []).length, 1);
  assert.match(html, /data-step="models"[^>]*disabled/);
  assert.match(html, /data-step="data" aria-label="[^"]*" aria-current="step"/);
  assert.ok(html.includes('<nav aria-label="Analysis steps"><ol class="steps">'));
});

test('privacy text for hosted mode is precise and never says nothing is uploaded', async () => {
  const t = await uploaded();
  const html = V.renderData(t.flow.getState());
  assert.match(html, /uploads the whole file to the service|stored by the service for up to 30 minutes without use/);
  assert.ok(!/nothing is uploaded|no data leaves|never leaves/i.test(html));
  assert.match(html, /30 minutes/);
  assert.match(html, /None of these downloads contains row-level data/);
  const before = V.renderData(newFlow().flow.getState());
  assert.match(before, /uploads the whole file to the service that serves this page/);
});

test('error summaries link to fields and are announced as alerts', async () => {
  const t = await uploaded();
  t.flow.next(); t.flow.next();
  const html = V.renderAnalyze(t.flow.getState(), t.flow);
  assert.match(html, /<div class="error-summary" id="error-summary" role="alert" tabindex="-1"><h3>There is a problem<\/h3>/);
  assert.match(html, /<a href="#confirm-target" data-action="focus-field" data-field="confirm-target">Confirm the target/);
  assert.match(html, /id="confirm-target"[^>]*aria-describedby="confirm-target-error"/);
});

test('describeError never loses the service message and never prints a code that is not in the wording table', () => {
  const m = FLOW.describeError(err(422, 'TABLE_TOO_MANY_ROWS', 'The table has 20,001 rows.'));
  assert.match(m, /could not use this input/); assert.match(m, /more rows than the service limit/); assert.match(m, /Service message: The table has 20,001 rows/);
  assert.match(FLOW.describeError(err(500, 'INTERNAL_ERROR', 'Internal.')), /could not complete the request/);
});

S.run('test_web_flow.js');
