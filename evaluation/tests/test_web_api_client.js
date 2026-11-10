/* Sherwan Ali. The API client: tokens travel only in the Authorization header, errors become readable objects,
   and hosted detection fails closed. Uses a fake fetch; no network. */
'use strict';
const assert = require('node:assert/strict');
const S = require('./web_support.js');
const { API } = S.load();
const { test } = S;

const TOKEN = 'TOKEN-c0ffee-1234567890abcdef';
function reply(status, body, headers = {}) {
  const h = new Map(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
  return { ok: status >= 200 && status < 300, status, headers: { get: k => h.get(String(k).toLowerCase()) ?? null },
    async json() { if (body === undefined || typeof body === 'string') throw Error('not json'); return body; }, async text() { return typeof body === 'string' ? body : JSON.stringify(body); }, async blob() { return { size: 3, type: 'x', body }; } };
}
function client(handler) {
  const seen = [];
  const fetchImpl = async (url, init) => { seen.push({ url, init }); return handler(url, init, seen.length); };
  return { api: API.createApiClient({ fetchImpl }), seen };
}

test('hosted detection needs a version answer and fails closed otherwise', async () => {
  assert.deepEqual(await client(() => reply(200, { app_version: '2', package_version: '0.3', report_schemas: {}, release_id: 'r' })).api.detectHosted(), { app_version: '2', package_version: '0.3', report_schemas: {}, release_id: 'r' });
  assert.ok(await client(() => reply(200, { app: '2', package: '0.3', schemas: {}, release_id: 'r' })).api.detectHosted(), 'the design wording of the field is accepted too');
  assert.equal(await client(() => reply(404, { error: 'Not found.' })).api.detectHosted(), null);
  assert.equal(await client(() => reply(200, { hello: 1 })).api.detectHosted(), null);
  assert.equal(await client(() => { throw new TypeError('network'); }).api.detectHosted(), null);
  assert.equal(await client(() => reply(200, '<html>static host page</html>')).api.detectHosted(), null);
  assert.equal(await client(() => reply(200, { ok: true, mode: 'local' })).api.detectLocal() !== null, true);
  assert.equal(await client(() => reply(200, { ok: true, mode: 'hosted' })).api.detectLocal(), null);
});

test('an upload sends raw bytes with an ASCII file name and no token', async () => {
  const { api, seen } = client(() => reply(201, { dataset_id: 'ds_1', capability: TOKEN }));
  const bytes = new Uint8Array([1, 2, 3]);
  const out = await api.createDataset(bytes, 'café data.csv');
  assert.equal(out.dataset_id, 'ds_1');
  const { url, init } = seen[0];
  assert.equal(url, '/api/v1/datasets'); assert.equal(init.method, 'POST'); assert.equal(init.body, bytes);
  assert.equal(init.headers['Content-Type'], 'application/octet-stream');
  assert.equal(init.headers['X-Filename'], 'caf_ data.csv');
  assert.ok(/^[\x20-\x7E]*$/.test(init.headers['X-Filename']));
  assert.ok(!('Authorization' in init.headers));
  assert.equal(init.cache, 'no-store'); assert.equal(init.redirect, 'error');
  assert.equal(API.headerFileName(''), 'upload.csv');
  assert.equal(API.headerFileName('a\r\nInjected: x.csv'), 'a  Injected: x.csv');
});

test('tokens go only in the Authorization header, never in a URL or a body', async () => {
  const { api, seen } = client((url) => url.endsWith('/result') ? reply(200, { schema_version: 'training-1.1' }) : url.includes('/exports/') ? reply(200, 'abc') : reply(200, { state: 'running' }));
  await api.createJob(TOKEN, { dataset_id: 'ds_1', target: 'y', positive_label: 'a' });
  await api.jobStatus('job_1', TOKEN); await api.cancelJob('job_1', TOKEN); await api.jobResult('job_1', TOKEN);
  await api.exportBlob('job_1', TOKEN, 'report_html'); await api.deleteDataset('ds_1', TOKEN);
  assert.equal(seen.length, 6);
  seen.forEach(({ url, init }) => {
    assert.ok(!url.includes(TOKEN), url);
    assert.equal(init.headers.Authorization, 'Bearer ' + TOKEN);
    assert.ok(!(typeof init.body === 'string' && init.body.includes(TOKEN)));
  });
  assert.deepEqual(seen.map(s => [s.init.method, s.url]), [['POST', '/api/v1/jobs'], ['GET', '/api/v1/jobs/job_1'], ['DELETE', '/api/v1/jobs/job_1'], ['GET', '/api/v1/jobs/job_1/result'], ['GET', '/api/v1/jobs/job_1/exports/report_html'], ['DELETE', '/api/v1/datasets/ds_1']]);
  assert.equal(seen[0].init.headers['Content-Type'], 'application/json');
});

test('ids and names are percent-encoded in paths', async () => {
  const { api, seen } = client(() => reply(200, 'a,b\n1,2\n'));
  await api.sampleText('liver disease&x=1.csv');
  assert.equal(seen[0].url, '/api/sample?name=liver%20disease%26x%3D1.csv');
  const c2 = client(() => reply(200, { state: 'x' }));
  await c2.api.jobStatus('a/b?c', TOKEN);
  assert.equal(c2.seen[0].url, '/api/v1/jobs/a%2Fb%3Fc');
});

test('a 202 result means the job is still live', async () => {
  const live = await client(() => reply(202, { state: 'running' })).api.jobResult('j', TOKEN);
  assert.equal(live.ready, false);
  const done = await client(() => reply(200, { schema_version: 'training-1.1' })).api.jobResult('j', TOKEN);
  assert.equal(done.ready, true);
});

test('errors become ApiError with code, message, details and Retry-After', async () => {
  const body = { error: { code: 'QUEUE_FULL', message: 'The service is busy with another job. Try again shortly.', details: { retry_after_seconds: 15 }, request_id: 'abc123' } };
  try { await client(() => reply(429, body, { 'Retry-After': '15' })).api.createJob(TOKEN, {}); assert.fail('should throw'); }
  catch (e) {
    assert.ok(e instanceof API.ApiError); assert.equal(e.status, 429); assert.equal(e.code, 'QUEUE_FULL'); assert.equal(e.retryAfter, 15); assert.equal(e.requestId, 'abc123'); assert.equal(e.kind, 'http');
    assert.ok(!JSON.stringify([e.message, e.details]).includes(TOKEN));
  }
  try { await client(() => reply(429, { error: { code: 'DATASET_BUSY', message: 'x', details: { retry_after_seconds: 7 } } })).api.createJob(TOKEN, {}); }
  catch (e) { assert.equal(e.retryAfter, 7, 'falls back to the detail value when the header is missing'); }
  try { await client(() => reply(400, { ok: false, error: 'Legacy text.' })).api.samples(); } catch (e) { assert.equal(e.message, 'Legacy text.'); assert.equal(e.code, null); }
  try { await client(() => reply(502, '<html>bad gateway</html>')).api.samples(); } catch (e) { assert.equal(e.status, 502); assert.match(e.message, /answered 502/); }
  try { await client(() => { throw new TypeError('Failed to fetch'); }).api.samples(); } catch (e) { assert.equal(e.kind, 'network'); assert.match(e.message, /could not be reached/); assert.ok(!e.message.includes('Failed to fetch')); }
  try { await client(() => reply(200, '<html></html>')).api.samples(); } catch (e) { assert.equal(e.kind, 'parse'); }
});

test('a missing fetch is reported as a network problem, not a crash', async () => {
  const api = API.createApiClient({ fetchImpl: null });
  const saved = globalThis.fetch; globalThis.fetch = undefined;
  try { const a = API.createApiClient({}); await assert.rejects(() => a.samples(), e => e.kind === 'network'); } finally { globalThis.fetch = saved; }
  await assert.rejects(() => api.samples(), e => e instanceof API.ApiError || e.kind === 'network');
});

test('no module other than the API client can reach the network or storage', () => {
  const offenders = [];
  S.webFiles().filter(f => f.endsWith('.js') && f !== 'api-client.js').forEach(f => {
    const src = S.readWeb(f);
    if (/\bfetch\s*\(|XMLHttpRequest|sendBeacon|WebSocket|EventSource|importScripts|new\s+Image\s*\(|navigator\.serviceWorker/.test(src)) offenders.push(f);
  });
  assert.deepEqual(offenders, []);
});

test('campaign context keeps only public labels and follows same-origin API calls', async () => {
  assert.deepEqual(API.campaignContext('?utm_source=cv&utm_medium=resume&utm_campaign=portfolio&utm_content=ml-engineer&email=private', ''),
    { utm_source: 'cv', utm_medium: 'resume', utm_campaign: 'portfolio', utm_content: 'ml-engineer' });
  assert.equal(API.campaignContext('?utm_source=private-person', ''), null);
  assert.deepEqual(API.campaignContext('', 'https://www.linkedin.com/feed/?private=value'), { utm_source: 'linkedin' });
  assert.equal(API.campaignContext('', 'https://linkedin.com.evil.test/'), null);
  const calls = [];
  const api = API.createApiClient({pageSearch: '?utm_source=cv&utm_content=python-backend', pageReferrer: '',
    fetchImpl: async (url,init) => { calls.push({url,init}); return reply(200,{app_version:'2'}); }});
  await api.detectHosted();
  assert.deepEqual(JSON.parse(calls[0].init.headers['X-FYD-Attribution']), {utm_source:'cv',utm_content:'python-backend'});
  assert.ok(!JSON.stringify(calls).includes('private'));
  const external = API.createApiClient({base:'https://different.test',pageSearch:'?utm_source=cv',
    fetchImpl: async (url,init) => { assert.ok(!init.headers['X-FYD-Attribution']); return reply(200,{app_version:'2'}); }});
  await external.detectHosted();
});

S.run('test_web_api_client.js');
