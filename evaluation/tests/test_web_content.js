/* Sherwan Ali. The research and method wording must stay inside what the repository's own documents say:
   every non-trivial number in the Research tab appears in README.md or research/RESEARCH-EVIDENCE.md, the two
   public links appear in the repository's own documents, and the separation between evidence sets is kept. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const S = require('./web_support.js');
const { C, V } = S.load();
const { test } = S;

const ROOT = S.REPO;                                       // repository root
const read = p => fs.readFileSync(path.join(ROOT, p), 'utf8');
const SOURCES = ['README.md', 'research/RESEARCH-EVIDENCE.md'].map(read).join('\n');
const ALL_DOCS = [SOURCES, ...['docs/LIMITATIONS_AND_VERIFICATION.md', 'docs/RESULTS_EVIDENCE.md', 'research/README.md', 'CITATION.cff'].map(read)].join('\n');

function strings(value, out = []) {
  if (typeof value === 'string') out.push(value);
  else if (Array.isArray(value)) value.forEach(v => strings(v, out));
  else if (value && typeof value === 'object') Object.values(value).forEach(v => strings(v, out));
  return out;
}
test('every decimal, percentage and comma-grouped number in the research wording appears in the repository documents', () => {
  const missing = [];
  strings(C.RESEARCH).forEach(s => {
    for (const m of s.matchAll(/[+-]?\d{1,3}(?:,\d{3})+|[+-]?\d+\.\d+%?|[+-]?\d+%/g)) {
      const token = m[0].replace(/^[+-]/, '');
      if (!SOURCES.includes(token)) missing.push(`${m[0]} in "${s.slice(0, 60)}"`);
    }
  });
  assert.deepEqual(missing, []);
});

test('the key counts are the documented ones', () => {
  const rows = C.RESEARCH.november.rows.map(r => [r.higher, r.tied, r.lower, r.difference]);
  assert.deepEqual(rows, [[16, 0, 4, '+0.0685'], [16, 0, 4, '+0.0093'], [12, 2, 6, '-0.0008']]);
  rows.forEach(r => assert.equal(r[0] + r[1] + r[2], 20, 'each row accounts for all 20 datasets'));
  assert.match(SOURCES, /16 of 20 datasets \(\+0\.0685/);
  assert.match(SOURCES, /higher on 12 datasets, tied on 2, lower on 6|ERF higher on 12 datasets, tied on 2, lower on 6|higher on 12, tied on 2 \(WINE, YEAST5ERL\), lower on 6/);
  assert.match(SOURCES, /0\.7872 against 0\.7880|0\.7872 for ERF against 0\.7880/);
  assert.match(SOURCES, /65\.7%/);
  assert.match(SOURCES, /480 cells planned: 382 newly successful, 96 reused and 2 failed/);
  assert.match(SOURCES, /42 of 143/);
  assert.match(SOURCES, /101 rows/);
  assert.match(SOURCES, /gate was not met/);
  assert.match(SOURCES, /5\.97/);
  const t = strings(C.RESEARCH).join(' ');
  assert.match(t, /2,000 successful fits/); assert.match(t, /not 2,000 datasets/);
  assert.match(t, /20 datasets are 18 distinct feature sets/);
  assert.match(t, /Not peer reviewed/);
  assert.match(t, /neither an advantage nor equivalence|shows neither an advantage nor equivalence/);
  assert.match(t, /no matched-compute study has been completed/);
  assert.match(t, /42 of 143/); assert.match(t, /say nothing about ERF/);
  assert.ok(!/state-of-the-art|peer-reviewed publication|independently confirmed|production adoption/i.test(t), 'no claim of publication or independent confirmation');
});

test('evidence sets stay separate: pilot, adaptive campaign, audit and the 20-dataset record each have their own heading and label', () => {
  const html = V.renderResearch(window.ERF_STUDY);
  ['Latest comparison: November 2026 record', 'Controlled pilot (separate protocol)', 'Adaptive campaign (separate procedure)', 'The CSRBoost audit (separate project)'].forEach(h => assert.ok(html.includes(`<h3>${h}</h3>`), h));
  assert.ok(html.indexOf('Latest comparison') < html.indexOf('Controlled pilot') && html.indexOf('Controlled pilot') < html.indexOf('Adaptive campaign') && html.indexOf('Adaptive campaign') < html.indexOf('The CSRBoost audit'));
  assert.ok((html.match(/historical research record/g) || []).length >= 4, 'each historical block is labelled');
  assert.match(html, /Counts are numbers of datasets\. They are not significance tests/);
  assert.match(html, /A result produced here is labelled a fresh app run, and it is never presented as a historical research record/);
  const audit = html.slice(html.indexOf('The CSRBoost audit'));
  assert.ok(!/ERF-CSRBoost mean|mean AP/.test(audit.split('Audit repository')[0]), 'audit text carries no ERF result');
  assert.equal(window.ERF_STUDY.planned_jobs, 144);
  assert.equal(window.ERF_STUDY.status_counts.success, 144);
  assert.equal(window.ERF_STUDY.confirmation_gate.met, false);
  assert.ok(Math.abs(window.ERF_STUDY.confirmation_gate.median_erf_to_balanced_rf_fit_ratio - 5.97) < 0.01);
});

test('the strengths come first on the research page, and the negative findings are stated plainly', () => {
  const html = V.renderResearch(window.ERF_STUDY);
  assert.ok(html.indexOf('higher on') > 0);
  assert.ok(html.indexOf('ERF-CSRBoost in brief') < html.indexOf('What limits these numbers'));
  const t = html.replace(/<[^>]+>/g, ' ');
  assert.match(t, /gate was not met/);
  assert.match(t, /does not establish an advantage/);
  assert.match(t, /no support for that step/);
  assert.match(t, /plain random forest had higher mean AP than ERF on 3 of 5 families/);
  assert.match(t, /cannot be rerun from the public repository/);
});

test('every link the page writes out appears verbatim in the repository documents', () => {
  [C.SOURCE_URL, C.AUDIT_URL, C.NOVEMBER_RECORD_URL].forEach(u => assert.ok(ALL_DOCS.includes(u), u));
  assert.equal(C.AUDIT_URL, 'https://github.com/SherwanAli0/csrboost-audit');
  const bare = ['https://github.com/SherwanAli0/erf-csrboost#'];
  const offenders = [];
  S.webFiles().filter(f => /\.(html|js)$/.test(f) && !f.startsWith('recorded-example')).forEach(f => {
    for (const m of S.readWeb(f).matchAll(/https:\/\/github\.com\/[A-Za-z0-9_.\/#-]+/g)) { if (!ALL_DOCS.includes(m[0]) || bare.includes(m[0])) offenders.push(`${f}: ${m[0]}`); }
  });
  assert.deepEqual(offenders, [], 'a URL that no README, CITATION or documentation line contains verbatim');
  const html = V.renderOverview({ hosted: false, preview: '' }) + V.renderResearch(window.ERF_STUDY);
  for (const m of html.matchAll(/href="(https:[^"]+)"/g)) assert.ok(ALL_DOCS.includes(m[1]), m[1]);
});

test('how-it-works statements agree with the package documents', () => {
  const rd = read('docs/RECIPE-DIFFERENCES.md');
  assert.match(rd, /200 trees, `class_weight="balanced"`/);
  assert.match(C.HOW.recipes.join(' '), /200-tree forest/);
  assert.match(rd, /101 fixed values \(0\.00 to 1\.00\)/);
  assert.match(C.HOW.steps.map(s => s.text).join(' '), /101 fixed thresholds from 0\.00 to 1\.00/);
  assert.match(rd, /maximizing F1 or minimizing assumed cost, highest value on ties/);
  assert.match(rd, /About 60 \/ 20 \/ 20|about 60 \/ 20 \/ 20/);
  assert.match(rd, /is not the research 100-tree raw forest/);
  const methods = fs.readFileSync(path.join(S.REPO, 'evaluation', 'docs', 'methods.md'), 'utf8');
  assert.ok(methods.length > 100);
  const api = read('docs/API.md');
  ['succeeded', 'partial', 'failed', 'cancelled', 'timed_out', 'interrupted', 'expired'].forEach(st => { assert.ok(api.includes('`' + st + '`'), st); assert.ok(C.JOB_STATE_TEXT[st], st); });
  ['parsing', 'profiling', 'splitting', 'fitting', 'evaluating', 'reporting', 'done'].forEach(st => { assert.ok(api.includes('`' + st + '`'), st); assert.ok(C.STAGE_TEXT[st], st); });
});

test('split question wording covers exactly the four package split modes', () => {
  const src = fs.readFileSync(path.join(S.REPO, 'evaluation', 'src', 'rare_event_eval', 'mixed.py'), 'utf8');
  const m = /SPLIT_MODES = \(([^)]*)\)/.exec(src)[1].match(/"(\w+)"/g).map(x => x.replace(/"/g, ''));
  assert.deepEqual(C.SPLIT_MODES.map(x => x.id).sort(), m.sort());
  C.SPLIT_MODES.forEach(x => { assert.ok(x.question.endsWith('?')); assert.ok(x.name && x.detail); });
});

test('recorded-example.js is an exact derived copy of recorded-example.json, so the example opens from a file: URL', () => {
  const json = path.join(S.WEB, 'recorded-example.json'), script = path.join(S.WEB, 'recorded-example.js');
  if (!fs.existsSync(json)) { assert.ok(!fs.existsSync(script), 'a derived copy without its source'); return; }
  assert.ok(fs.existsSync(script), 'run: node evaluation/tests/sync_recorded_example.js');
  const context = { window: {} };
  require('node:vm').runInNewContext(fs.readFileSync(script, 'utf8'), context);
  assert.deepEqual(JSON.parse(JSON.stringify(context.window.FYD_RECORDED_EXAMPLE)), JSON.parse(fs.readFileSync(json, 'utf8')), 'out of date: run node evaluation/tests/sync_recorded_example.js');
});

test('every reason code the service or the package can report has plain wording or is shown with the service message', () => {
  const doc = read('evaluation/docs/report-contract.md');
  const listed = ['MODEL_TIMEOUT', 'WORKER_NO_RESULT', 'MODEL_UNKNOWN', 'MODEL_DEPENDENCY_MISSING', 'CALIBRATION_INFEASIBLE', 'RESAMPLING_INFEASIBLE', 'SPLIT_INFEASIBLE', 'INPUT_INVALID', 'MODEL_OUT_OF_MEMORY', 'MODEL_FIT_ERROR'];
  listed.forEach(code => { assert.ok(doc.includes(code), code + ' is documented'); assert.ok(C.REASONS[code], code + ' has wording'); });
  ['JOB_TIMEOUT', 'QUEUE_WAIT_EXCEEDED', 'CANCELLED', 'SERVICE_STOPPING', 'MEMORY_LIMIT_EXCEEDED', 'RESULT_TOO_LARGE', 'JOB_CRASHED', 'SOME_MODELS_FAILED'].forEach(code => assert.ok(C.REASONS[code], code));
  Object.values(C.REASONS).forEach(t => assert.ok(!/[<>]/.test(t)));
});

S.run('test_web_content.js');
