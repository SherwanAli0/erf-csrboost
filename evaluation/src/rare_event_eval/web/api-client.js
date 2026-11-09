/* Sherwan Ali. The only place in the workbench that talks to a server. Access tokens ("capabilities") are sent in
   the Authorization header and nowhere else: never in a URL, never to storage, never to the console. */
(function (root) {
  'use strict';
  class ApiError extends Error {
    constructor(info) {
      super(info.message || 'The request failed.');
      this.name = 'ApiError';
      this.kind = info.kind || 'http';          // http | network | timeout | parse
      this.status = info.status || 0;
      this.code = info.code || null;
      this.details = info.details || {};
      this.retryAfter = typeof info.retryAfter === 'number' ? info.retryAfter : null;
      this.requestId = info.requestId || null;
    }
  }
  /* Header values must be Latin-1 for fetch; the service only reads the extension, so keep it ASCII. */
  function headerFileName(name) {
    const text = String(name == null ? '' : name).replace(/[\r\n]/g, ' ');
    const clean = text.replace(/[^\x20-\x7E]/g, '_').slice(0, 200);
    return clean || 'upload.csv';
  }
  function parseRetryAfter(headers, details) {
    const raw = headers && typeof headers.get === 'function' ? headers.get('Retry-After') : null;
    const n = raw == null ? NaN : Number(raw);
    if (Number.isFinite(n) && n >= 0) return n;
    const d = details && details.retry_after_seconds;
    return typeof d === 'number' && Number.isFinite(d) && d >= 0 ? d : null;
  }
  async function toError(response) {
    let body = null;
    try { body = await response.json(); } catch (_) { body = null; }
    const err = body && body.error;
    let code = null, message = null, details = {}, requestId = null;
    if (err && typeof err === 'object') { code = err.code || null; message = err.message || null; details = err.details || {}; requestId = err.request_id || null; }
    else if (typeof err === 'string') message = err;          // legacy routes: {"ok": false, "error": "text"}
    return new ApiError({ kind: 'http', status: response.status, code, message: message || `The service answered ${response.status}.`, details,
      retryAfter: parseRetryAfter(response.headers, details), requestId });
  }

  function createApiClient(options = {}) {
    const fetchImpl = options.fetchImpl || (typeof globalThis.fetch === 'function' ? globalThis.fetch.bind(globalThis) : null);
    const base = options.base || '';
    async function request(method, path, opts = {}) {
      if (!fetchImpl) throw new ApiError({ kind: 'network', message: 'This browser cannot make network requests.' });
      const headers = Object.assign({}, opts.headers || {});
      if (opts.bearer) headers.Authorization = 'Bearer ' + opts.bearer;
      const init = { method, headers, cache: 'no-store', credentials: 'same-origin', redirect: 'error' };
      if (opts.body !== undefined) init.body = opts.body;
      if (opts.signal) init.signal = opts.signal;
      let timer = null;
      if (opts.timeoutMs && typeof AbortController === 'function') { const ctl = new AbortController(); init.signal = ctl.signal; timer = setTimeout(() => ctl.abort(), opts.timeoutMs); }
      let response;
      try { response = await fetchImpl(base + path, init); }
      catch (e) {
        throw new ApiError({ kind: e && e.name === 'AbortError' ? 'timeout' : 'network', message: e && e.name === 'AbortError' ? 'The service did not answer in time.' : 'The service could not be reached. Check your connection and try again.' });
      } finally { if (timer) clearTimeout(timer); }
      if (!response.ok) throw await toError(response);
      return response;
    }
    async function json(method, path, opts) {
      const response = await request(method, path, opts);
      try { return { status: response.status, data: await response.json(), headers: response.headers }; }
      catch (_) { throw new ApiError({ kind: 'parse', status: response.status, message: 'The service answer could not be read.' }); }
    }
    const enc = encodeURIComponent;
    return {
      /* Hosted detection: the version route answers only on the hosted service. Any failure means "not hosted". */
      async detectHosted() {
        try { const r = await json('GET', '/api/v1/version', { timeoutMs: 2500 }); const d = r.data;
          return d && typeof d === 'object' && ('app_version' in d || 'app' in d) ? d : null; }
        catch (_) { return null; }
      },
      async detectLocal() {
        try { const r = await json('GET', '/api/health', { timeoutMs: 2000 }); return r.data && r.data.mode === 'local' ? r.data : null; }
        catch (_) { return null; }
      },
      async samples() { return (await json('GET', '/api/v1/samples')).data; },
      async dataTool(operation, body, signal) {
        return (await json('POST', '/api/v1/tools/' + enc(operation), {
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal
        })).data;
      },
      async sampleText(name) { return (await request('GET', '/api/sample?name=' + enc(name))).text(); },
      /* Standalone local server only: the older numeric-feature training route. */
      async trainLocal(body) { return (await json('POST', '/api/train', { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).data; },
      async staticJson(path) { return (await json('GET', path)).data; },
      async createDataset(bytes, filename, selection) {
        const headers = { 'Content-Type': 'application/octet-stream', 'X-Filename': headerFileName(filename) };
        if (selection && Object.keys(selection).length) headers['X-Import-Selection']=encodeURIComponent(JSON.stringify(selection));
        return (await json('POST', '/api/v1/datasets', { headers, body: bytes })).data;
      },
      async deleteDataset(id, cap) { return (await json('DELETE', '/api/v1/datasets/' + enc(id), { bearer: cap })).data; },
      async createJob(cap, body) {
        return (await json('POST', '/api/v1/jobs', { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), bearer: cap })).data;
      },
      async jobStatus(id, cap) { return (await json('GET', '/api/v1/jobs/' + enc(id), { bearer: cap })).data; },
      async cancelJob(id, cap) { return (await json('DELETE', '/api/v1/jobs/' + enc(id), { bearer: cap })).data; },
      /* 200 returns the report; 202 means the job is still live and returns its status document. */
      async jobResult(id, cap) {
        const r = await json('GET', '/api/v1/jobs/' + enc(id) + '/result', { bearer: cap });
        return { ready: r.status === 200, data: r.data };
      },
      async exportBlob(id, cap, kind) {
        const response = await request('GET', '/api/v1/jobs/' + enc(id) + '/exports/' + enc(kind), { bearer: cap });
        return response.blob();
      }
    };
  }
  const api = { createApiClient, ApiError, headerFileName };
  root.FYD = root.FYD || {};
  root.FYD.api = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
