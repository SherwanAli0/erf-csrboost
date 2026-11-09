/* Sherwan Ali. Two failures that really happen: a page script that does not load, and a samples request that fails.
   The page must show what went wrong and let the person try again. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const S = require('./web_support.js');
const { FLOW, V, API } = S.load();
const PRE = require(path.join(S.WEB, 'preflight.js'));
const { test } = S;

/* A stand-in for the page's global object holding what each script would have set. */
function fullRoot() {
  return { RareEventMetrics: globalThis.RareEventMetrics, ERF_EXAMPLES: globalThis.ERF_EXAMPLES, ERF_TRAINING_CSV: globalThis.ERF_TRAINING_CSV, FYD: Object.assign({}, globalThis.FYD), FYD_BOOTED: true };
}
const OWNER = { 'scoring-core.js': r => { delete r.RareEventMetrics; }, 'demo-data.js': r => { delete r.ERF_EXAMPLES; }, 'util.js': r => { delete r.FYD.util; }, 'content.js': r => { delete r.FYD.content; },
  'models.js': r => { delete r.FYD.MODELS; }, 'report-view.js': r => { delete r.FYD.reportView; }, 'predictions.js': r => { delete r.FYD.predictions; }, 'api-client.js': r => { delete r.FYD.api; },
  'flow.js': r => { delete r.FYD.flow; }, 'views.js': r => { delete r.FYD.views; }, 'tools.js': r => { delete r.FYD.tools; } };

test('nothing is reported missing when every script ran', () => {
  assert.deepEqual(PRE.missing(fullRoot(), true), []);
});

test('each required script is detected by name when its global is missing', () => {
  for (const [script, remove] of Object.entries(OWNER)) {
    const r = fullRoot(); remove(r);
    assert.deepEqual(PRE.missing(r, true), [script], script);
  }
  const r = fullRoot(); delete r.FYD_BOOTED;
  assert.deepEqual(PRE.missing(r, true), ['app.js']);
  assert.deepEqual(PRE.missing({}, true).length, Object.keys(OWNER).length + 1, 'an empty page reports everything');
  const broken = fullRoot(); broken.FYD = null;
  assert.ok(PRE.missing(broken, false).length >= 6, 'a missing namespace is not a crash');
});

test('the check covers every deferred script the page depends on', () => {
  const html = S.readWeb('index.html');
  const loaded = [...html.matchAll(/<script defer src="([^"]+)"><\/script>/g)].map(m => m[1]);
  const optional = ['preflight.js', 'app.js', 'recorded-example.js'];
  loaded.filter(f => !optional.includes(f)).forEach(f => assert.ok(PRE.REQUIRED_SCRIPTS.includes(f), f + ' is not checked'));
  PRE.REQUIRED_SCRIPTS.forEach(f => assert.ok(loaded.includes(f), f + ' is checked but not loaded'));
  assert.equal(loaded[0], 'preflight.js', 'loaded first');
  assert.equal(loaded[loaded.length - 1], 'app.js');
  assert.ok(Object.keys(OWNER).every(f => PRE.REQUIRED_SCRIPTS.includes(f)));
});

function fakePage() {
  const main = { html: '', insertAdjacentHTML(pos, html) { assert.equal(pos, 'afterbegin'); this.html = html + this.html; } };
  const button = { handlers: {}, addEventListener(type, fn) { this.handlers[type] = fn; } };
  const reloads = [];
  const doc = { body: {}, getElementById(id) { return id === 'main' ? main : id === 'boot-banner' ? (main.html.includes('id="boot-banner"') ? {} : null) : id === 'reload-page' ? (main.html ? button : null) : null; } };
  return { main, button, reloads, root: Object.assign(fullRoot(), { document: doc, location: { reload() { reloads.push(1); } } }) };
}

test('a missing script puts a visible alert with a Reload button above the panels, once', () => {
  const page = fakePage(); delete page.root.RareEventMetrics;
  const gone = PRE.install(page.root);
  assert.deepEqual(gone, ['scoring-core.js']);
  assert.match(page.main.html, /^<div class="notice boot-banner" id="boot-banner" role="alert">/);
  assert.ok(page.main.html.includes('Part of this page did not load. Reload the page to try again.'));
  assert.match(page.main.html, /<button type="button" class="btn" id="reload-page">Reload<\/button>/);
  assert.deepEqual(S.tagAudit(PRE.bannerHtml()), []);
  assert.equal(page.reloads.length, 0);
  page.button.handlers.click();
  assert.equal(page.reloads.length, 1, 'the button reloads the page');
  PRE.install(page.root);
  assert.equal((page.main.html.match(/id="boot-banner"/g) || []).length, 1, 'not duplicated');
});

test('no alert is shown when everything loaded', () => {
  const page = fakePage();
  assert.deepEqual(PRE.install(page.root), []);
  assert.equal(page.main.html, '');
});

test('app.js stops before it touches a dependency when one is missing, and the alert text is in the markup file set', () => {
  const src = S.readWeb('app.js');
  const ctx = { FYD_PREFLIGHT: PRE, FYD: { util: {} } };
  ctx.window = ctx;
  assert.doesNotThrow(() => vm.runInNewContext(src, ctx));       // the original failure was "Cannot read properties of undefined (reading 'fromCSV')"
  assert.equal(ctx.FYD_BOOTED, undefined, 'it did not start');
  const without = { FYD: {} }; without.window = without;
  assert.throws(() => vm.runInNewContext(src, without), 'without the preflight guard the same page would crash');
  assert.match(src, /window\.FYD_BOOTED = true;\n\s+boot\(\);/);
  assert.match(S.readWeb('preflight.js'), /Part of this page did not load\. Reload the page to try again\./);
});

/* ---- a failing samples request ---- */
const SAMPLES = { samples: [{ name: 'a.csv', category: 'Medical', label: 'Alpha', desc: 'x' }, { name: 'b.csv', category: 'Finance', label: 'Beta', desc: 'y' }], order: ['Medical', 'Finance'] };
function flowWith(results) {
  const queue = [...results], calls = [];
  const api = { async samples() { calls.push('samples'); const r = queue.shift(); if (r instanceof Error) throw r; return r; } };
  return { flow: FLOW.createAnalysisFlow({ api, schedule: () => 0, unschedule: () => {} }), calls };
}
const down = () => new API.ApiError({ kind: 'network', message: 'The service could not be reached. Check your connection and try again.' });

test('a failed samples request shows a Try again button that refetches, announces the result and moves focus', async () => {
  const { flow, calls } = flowWith([down(), down(), SAMPLES]);
  await flow.loadSamples();
  let st = flow.getState();
  assert.equal(st.samples.phase, 'error'); assert.equal(st.focus.n, 0, 'the first load never moves focus');
  let html = V.renderData(st);
  assert.match(html, /The sample list could not be loaded\. The service could not be reached/);
  assert.match(html, /<button type="button" class="btn" id="retry-samples" data-action="retry-samples">Try again<\/button>/);
  assert.deepEqual(S.tagAudit(html), []);
  // second failure: stays retryable, says so, focus returns to the button
  await flow.retrySamples();
  st = flow.getState();
  assert.equal(st.samples.phase, 'error'); assert.equal(st.status, 'The sample list could not be loaded. Try again.');
  assert.equal(st.focus.id, 'retry-samples'); assert.equal(st.focus.n, 1);
  assert.match(V.renderData(st), /id="retry-samples"/);
  // success: list shown, announced, focus to the first sample
  await flow.retrySamples();
  st = flow.getState();
  assert.equal(st.samples.phase, 'ready'); assert.equal(st.samples.items.length, 2);
  assert.equal(st.status, 'The sample list loaded: 2 samples.');
  assert.equal(st.focus.id, 'sample-a.csv'); assert.equal(st.focus.n, 2);
  html = V.renderData(st);
  assert.ok(!html.includes('retry-samples') && html.includes('id="sample-a.csv"'));
  assert.deepEqual(calls, ['samples', 'samples', 'samples']);
});

test('while the list is loading a second retry does nothing, and a ready list is not refetched by the first-load path', async () => {
  const { flow, calls } = flowWith([SAMPLES, SAMPLES]);
  const first = flow.loadSamples(); const again = flow.loadSamples(true);
  await Promise.all([first, again]);
  assert.equal(calls.length, 1);
  await flow.loadSamples();
  assert.equal(calls.length, 1);
});

test('the Try again button is wired in the page glue', () => {
  assert.match(S.readWeb('app.js'), /a === 'retry-samples'\) flow\.retrySamples\(\)/);
});

S.run('test_web_resilience.js');
