/* Sherwan Ali. Shared loader and a tiny test runner for the browser-module tests. No dependencies, Node 18 or newer.
   The browser modules attach to globalThis, so loading them here gives the same objects the page uses. */
'use strict';
const path = require('node:path');
const fs = require('node:fs');
const WEB = path.join(__dirname, '..', 'src', 'rare_event_eval', 'web');
const REPO = path.join(__dirname, '..', '..');

function load() {
  globalThis.window = globalThis;
  const metrics = require(path.join(WEB, 'scoring-core.js'));
  require(path.join(WEB, 'demo-data.js'));
  const U = require(path.join(WEB, 'util.js'));
  const C = require(path.join(WEB, 'content.js'));
  const MODELS = require(path.join(WEB, 'models.js'));
  const RV = require(path.join(WEB, 'report-view.js'));
  const P = require(path.join(WEB, 'predictions.js'));
  const API = require(path.join(WEB, 'api-client.js'));
  const FLOW = require(path.join(WEB, 'flow.js'));
  const V = require(path.join(WEB, 'views.js'));
  const TOOLS = require(path.join(WEB, 'tools.js'));
  return { M: metrics, U, C, MODELS, RV, P, API, FLOW, V, TOOLS };
}

const cases = [];
function test(name, fn) { cases.push({ name, fn }); }
async function run(file) {
  let failed = 0;
  for (const c of cases) {
    try { await c.fn(); } catch (e) { failed++; console.error(`FAIL ${c.name}\n  ${e && e.stack ? e.stack.split('\n').slice(0, 6).join('\n  ') : e}`); }
  }
  console.log(JSON.stringify({ file, tests: cases.length, failed, status: failed ? 'failed' : 'passed' }));
  process.exitCode = failed ? 1 : 0;
}
function readWeb(name) { return fs.readFileSync(path.join(WEB, name), 'utf8'); }
function webFiles() { return fs.readdirSync(WEB).filter(f => fs.statSync(path.join(WEB, f)).isFile()); }
function fixture(name) { return JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf8')); }

/* Scan generated HTML. Every tag must be on the allowlist, no attribute may be an event handler or a style, no
   href or src may use a script scheme, and a raw "<" may only start a tag. Returns a list of problems. */
const BASE_TAGS = new Set(['div', 'p', 'span', 'strong', 'em', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'table', 'caption',
  'thead', 'tbody', 'tr', 'th', 'td', 'details', 'summary', 'figure', 'figcaption', 'svg', 'rect', 'a', 'nav', 'button', 'input', 'select', 'option', 'label',
  'fieldset', 'legend', 'aside', 'br', 'code', 'pre']);
function tagAudit(html, extraTags = []) {
  const allowed = new Set([...BASE_TAGS, ...extraTags]), problems = [];
  let i = 0;
  while ((i = html.indexOf('<', i)) !== -1) {
    const m = /^<(\/?)([A-Za-z][A-Za-z0-9-]*)/.exec(html.slice(i, i + 40));
    if (!m) { problems.push('stray < at ' + i + ': ' + html.slice(i, i + 30)); i++; continue; }
    const name = m[2].toLowerCase();
    if (!allowed.has(name)) problems.push('tag not allowed: ' + name);
    let j = i + m[0].length;
    if (!m[1]) {
      while (j < html.length && html[j] !== '>') {
        while (j < html.length && (/\s/.test(html[j]) || html[j] === '/')) j++;
        if (html[j] === '>') break;
        const a = /^[^\s=>/]+/.exec(html.slice(j));
        if (!a) { problems.push('bad attribute near ' + html.slice(j, j + 30)); break; }
        const attr = a[0].toLowerCase(); j += a[0].length;
        if (/^on/.test(attr) || attr === 'style' || attr === 'srcdoc') problems.push('forbidden attribute ' + attr + ' on ' + name);
        let value = '';
        if (html[j] === '=') {
          j++;
          if (html[j] === '"' || html[j] === "'") {
            const q = html[j], end = html.indexOf(q, j + 1);
            if (end < 0) { problems.push('unterminated attribute value'); break; }
            value = html.slice(j + 1, end); j = end + 1;
          } else { const v = /^[^\s>]+/.exec(html.slice(j)); value = v ? v[0] : ''; j += value.length; }
        }
        if ((attr === 'href' || attr === 'src') && /^\s*(javascript|data|vbscript):/i.test(value)) problems.push('scheme in ' + attr + ': ' + value.slice(0, 20));
        if (value.includes('<') || value.includes('>')) problems.push('raw angle bracket in attribute ' + attr);
      }
    } else { while (j < html.length && html[j] !== '>') j++; }
    i = j + 1;
  }
  return problems;
}
function duplicateIds(html) {
  const seen = new Map(), dup = [];
  for (const m of html.matchAll(/\sid="([^"]*)"/g)) seen.set(m[1], (seen.get(m[1]) || 0) + 1);
  seen.forEach((n, id) => { if (n > 1) dup.push(id); });
  return dup;
}
module.exports = { load, test, run, readWeb, webFiles, fixture, tagAudit, duplicateIds, WEB, REPO };
