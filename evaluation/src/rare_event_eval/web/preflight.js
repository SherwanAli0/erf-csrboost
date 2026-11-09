/* Sherwan Ali. Loaded first. A page script can fail to load (a request can be refused or dropped), and then the page
   would be a blank shell. After every deferred script has run, this checks that each one left its global behind and that
   app.js started. If not, a visible alert with a Reload button is placed above the panels and the rest of the shell stays
   usable. app.js also calls missing() before it touches any dependency. */
(function (root) {
  'use strict';
  /* script file -> a test that says it ran. recorded-example.js is optional (the example is also fetched), so it is not listed. */
  const REQUIRED = [
    ['scoring-core.js', r => r.RareEventMetrics && typeof r.RareEventMetrics.evaluate === 'function'],
    ['demo-data.js', r => Array.isArray(r.ERF_EXAMPLES) && typeof r.ERF_TRAINING_CSV === 'string'],
    ['util.js', r => r.FYD && r.FYD.util],
    ['content.js', r => r.FYD && r.FYD.content],
    ['models.js', r => r.FYD && Array.isArray(r.FYD.MODELS)],
    ['report-view.js', r => r.FYD && r.FYD.reportView],
    ['predictions.js', r => r.FYD && r.FYD.predictions],
    ['api-client.js', r => r.FYD && r.FYD.api],
    ['flow.js', r => r.FYD && r.FYD.flow],
    ['views.js', r => r.FYD && r.FYD.views],
    ['tools.js', r => r.FYD && r.FYD.tools]
  ];
  /* Names of the scripts whose global is missing. With includeApp, app.js is checked as well (it sets FYD_BOOTED when it starts). */
  function missing(r, includeApp) {
    const out = REQUIRED.filter(([, ok]) => { try { return !ok(r); } catch (_) { return true; } }).map(([name]) => name);
    if (includeApp && !r.FYD_BOOTED) out.push('app.js');
    return out;
  }
  const MESSAGE = 'Part of this page did not load. Reload the page to try again.';
  const bannerHtml = () => `<div class="notice boot-banner" id="boot-banner" role="alert"><p>${MESSAGE}</p><p><button type="button" class="btn" id="reload-page">Reload</button></p></div>`;
  function install(r) {
    const gone = missing(r, true);
    if (!gone.length || !r.document) return gone;
    const main = r.document.getElementById('main') || r.document.body;
    if (!r.document.getElementById('boot-banner')) main.insertAdjacentHTML('afterbegin', bannerHtml());
    const button = r.document.getElementById('reload-page');
    if (button) button.addEventListener('click', () => r.location.reload());
    return gone;
  }
  function start(r) {
    if (!r.document) return;
    if (r.document.readyState === 'complete') install(r); else r.addEventListener('DOMContentLoaded', () => install(r));
  }
  const api = { missing, bannerHtml, install, MESSAGE, REQUIRED_SCRIPTS: REQUIRED.map(x => x[0]) };
  root.FYD_PREFLIGHT = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else start(root);
})(globalThis);
