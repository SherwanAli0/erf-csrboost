/* Sherwan Ali. Static checks of the web folder: contrast by calculation for every colour pair, ASCII text, no external
   asset, no inline script or style, no storage or console use, wiring between markup and handlers, and landmarks.
   Set FYD_PRINT_CONTRAST=1 to print the contrast table. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const S = require('./web_support.js');
const { test } = S;

const css = S.readWeb('style.css'), html = S.readWeb('index.html');
const jsFiles = S.webFiles().filter(f => f.endsWith('.js'));
const TEXT_FILES = S.webFiles().filter(f => /\.(html|css|js|json|svg|txt|md)$/.test(f));

/* ---------- contrast ---------- */
function tokens() {
  const root = /:root\s*{([^}]*)}/.exec(css)[1], out = {};
  for (const m of root.matchAll(/--([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;/g)) out[m[1]] = m[2].toLowerCase();
  return out;
}
const lin = c => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const lum = hex => { const n = parseInt(hex.slice(1), 16); return 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255); };
const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
const T = tokens();
const SURFACES = ['bg', 'surface', 'surface-2'];
const GENERAL_TEXT = ['text', 'text-muted', 'text-link', 'text-brand'];
const PAIRS = [
  ...GENERAL_TEXT.flatMap(t => SURFACES.map(s => [t, s, 4.5, 'text'])),
  ['on-primary', 'primary', 4.5, 'text'], ['on-danger', 'danger', 4.5, 'text'], ['on-warn', 'warn-bg', 4.5, 'text'], ['on-ok', 'ok-bg', 4.5, 'text'],
  ['on-bad', 'bad-bg', 4.5, 'text'], ['on-bad', 'surface', 4.5, 'text'], ['on-badge', 'badge-bg', 4.5, 'text'], ['text', 'warn-bg', 4.5, 'text'], ['text', 'ok-bg', 4.5, 'text'], ['text', 'bad-bg', 4.5, 'text'],
  ['line-ui', 'surface', 3, 'ui border'], ['line-ui', 'bg', 3, 'ui border'], ['line-ui', 'surface-2', 3, 'ui border'], ['focus', 'bg', 3, 'focus ring'], ['focus', 'surface', 3, 'focus ring'], ['focus', 'surface-2', 3, 'focus ring'],
  ['focus', 'warn-bg', 3, 'focus ring'], ['primary', 'bg', 3, 'ui accent'], ['primary', 'surface', 3, 'ui accent'], ['danger', 'surface', 3, 'invalid border'], ['danger', 'bg', 3, 'invalid border'],
  ['bar-fill', 'bar-track', 3, 'chart bar'], ['bar-fill', 'surface', 3, 'chart bar'], ['bar-base', 'bar-track', 3, 'chart baseline'], ['bar-base', 'surface', 3, 'chart baseline and its halo']
];
test('every colour token is a six-digit hex value and every pair meets its threshold', () => {
  Object.entries(T).forEach(([k, v]) => assert.match(v, /^#[0-9a-f]{6}$/, k));
  const failures = [];
  const rows = PAIRS.map(([fg, bg, min, use]) => {
    assert.ok(T[fg] && T[bg], `unknown token in pair ${fg} on ${bg}`);
    const r = ratio(T[fg], T[bg]);
    if (r < min) failures.push(`${fg} ${T[fg]} on ${bg} ${T[bg]}: ${r.toFixed(2)} < ${min}`);
    return `| ${use} | --${fg} ${T[fg]} | --${bg} ${T[bg]} | ${r.toFixed(2)} | ${min} | ${r >= min ? 'pass' : 'FAIL'} |`;
  });
  if (process.env.FYD_PRINT_CONTRAST) console.log(['| Use | Foreground | Background | Ratio | Minimum | Result |', '|---|---|---|---:|---:|---|', ...rows].join('\n'));
  assert.deepEqual(failures, []);
});

function rules() {
  const stripped = css.replace(/\/\*[\s\S]*?\*\//g, '');
  const list = [];
  for (const m of stripped.matchAll(/([^{}]+)\{([^{}]*)\}/g)) list.push({ selector: m[1].trim().replace(/\s+/g, ' '), body: m[2] });
  return list;
}
const decl = (body, prop) => { const m = new RegExp('(?:^|;)\\s*' + prop + '\\s*:\\s*([^;]+)', 'i').exec(body); return m ? m[1].trim() : null; };
const varName = v => { const m = v && /^var\(--([a-z0-9-]+)\)$/.exec(v); return m ? m[1] : null; };
test('every colour declaration uses a token, and each text and background pair in one rule is a checked pair', () => {
  const root = /:root\s*{[^}]*}/.exec(css)[0];
  const rest = css.replace(root, '');
  assert.ok(!/#[0-9a-fA-F]{3,8}\b/.test(rest.replace(/\/\*[\s\S]*?\*\//g, '')), 'hex colours only on :root');
  assert.ok(!/\b(rgb|rgba|hsl|hsla)\(/.test(rest), 'no other colour functions');
  const pairSet = new Set(PAIRS.map(p => p[0] + '|' + p[1]));
  const problems = [];
  rules().forEach(({ selector, body }) => {
    if (selector.startsWith(':root')) return;
    const color = decl(body, 'color'), bg = decl(body, 'background') || decl(body, 'background-color');
    const fg = varName(color), back = varName(bg);
    if (color && !fg && !/^(inherit|transparent|currentcolor)$/i.test(color)) problems.push(`${selector}: colour is not a token: ${color}`);
    if (bg && !back && !/^(none|transparent|inherit)$/i.test(bg)) problems.push(`${selector}: background is not a token: ${bg}`);
    if (fg && back && !pairSet.has(fg + '|' + back)) problems.push(`${selector}: pair ${fg} on ${back} is not in the checked list`);
    if (fg && !back && !GENERAL_TEXT.includes(fg) && !['.step.is-current .step-state', '.status-ok', '.status-bad', '.field-error'].includes(selector) && !fg.startsWith('on-')) problems.push(`${selector}: ${fg} may sit on any surface`);
    if (fg && fg.startsWith('on-') && !back && !['.step.is-current .step-state', '.error-summary h3'].includes(selector)) problems.push(`${selector}: ${fg} without its background in the same rule`);
    ['fill', 'border-color', 'border-left-color', 'border-bottom-color', 'outline-color', 'accent-color'].forEach(p => { const v = decl(body, p); if (v && !varName(v) && !/^(transparent|currentcolor|inherit|none)$/i.test(v)) problems.push(`${selector}: ${p} is not a token`); });
  });
  assert.deepEqual(problems, []);
});

test('the stylesheet supports keyboard focus, reduced motion, 44 px targets and small screens', () => {
  assert.match(css, /:focus-visible\s*{[^}]*outline:\s*3px solid var\(--focus\)/);
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)[^{]*{[^}]*animation: none !important[^}]*transition: none !important/);
  assert.match(css, /\.btn\s*{[^}]*min-height: 44px/);
  assert.match(css, /input\[type="text"\][^{]*{[^}]*min-height: 44px/);
  assert.match(css, /\.table-scroll\s*{[^}]*overflow-x: auto[^}]*max-width: 100%/);
  assert.match(css, /@media \(max-width: 560px\)/);
  assert.match(css, /color-scheme: light/);
  assert.match(css, /\.visually-hidden/);
  assert.ok(!/prefers-color-scheme/.test(css), 'one theme is supported, and that theme is the one that was checked');
  assert.ok(!/body\s*{[^}]*overflow-x:\s*(scroll|auto)/.test(css));
  assert.ok(!/(^|[;{\s])width:\s*\d{4,}px/.test(css), 'no fixed wide widths');
  assert.ok(!/@import|url\(\s*['"]?https?:/.test(css));
});

/* ---------- text and assets ---------- */
test('web files are plain ASCII: no em dash, no en dash, no smart quotes', () => {
  const bad = [];
  TEXT_FILES.forEach(f => { const buf = fs.readFileSync(path.join(S.WEB, f)); for (let i = 0; i < buf.length; i++) if (buf[i] > 127) { bad.push(`${f} byte ${i}`); break; } });
  assert.deepEqual(bad, []);
});

test('no external asset: only relative scripts, styles and images; links go to the two public repositories', () => {
  const ALLOWED = [/^https:\/\/github\.com\/SherwanAli0\/(erf-csrboost|csrboost-audit)(\/|#|$|")/, /^http:\/\/127\.0\.0\.1:8601/];
  const offenders = [];
  TEXT_FILES.forEach(f => { const src = S.readWeb(f); for (const m of src.matchAll(/https?:\/\/[^\s'"<>)`]+/g)) { if (!ALLOWED.some(r => r.test(m[0] + '"'))) offenders.push(`${f}: ${m[0]}`); } });
  assert.deepEqual(offenders, []);
  for (const m of html.matchAll(/<(script|link|img)\b[^>]*>/g)) { const url = /(?:src|href)="([^"]*)"/.exec(m[0]); if (url) assert.ok(!/^(https?:)?\/\//i.test(url[1]) && (url[1] === 'data:,' || !/^data:/i.test(url[1])), m[0]); }
  const views = S.readWeb('views.js');
  for (const m of views.matchAll(/<img\b[^>]*>/g)) { assert.ok(/ alt="[^"]+"/.test(m[0]), 'an image has alt text'); assert.ok(!/src="(https?:)?\/\//.test(m[0])); }
  assert.ok(!/@font-face|fonts\.googleapis|cdn\./i.test(css + html));
});

test('no inline script, style or event handler in the markup or in the templates', () => {
  assert.ok(!/<script(?![^>]*\bsrc=)[^>]*>/i.test(html), 'only external script files');
  const files = ['index.html', ...jsFiles];
  const offenders = [];
  files.forEach(f => { const src = S.readWeb(f); if (/\sstyle\s*=\s*["']/.test(src) || /\son[a-z]+\s*=\s*["']/.test(src) || /javascript:/i.test(src.replace(/payload|test/g, '')) || /\.setAttribute\(\s*['"]on/.test(src)) offenders.push(f); });
  assert.deepEqual(offenders, [], 'a page with a content security policy of default-src self must still work');
});

test('no storage, cookie, console, eval or history use; tokens cannot reach a URL or storage', () => {
  const offenders = [];
  jsFiles.forEach(f => { const src = S.readWeb(f).replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/[^\n]*/g, '$1'); if (/localStorage|sessionStorage|indexedDB|document\.cookie|console\.|\beval\s*\(|new Function|document\.write|history\.(push|replace)State|location\.(assign|replace)|location\.href\s*=|location\.search\s*=/.test(src)) offenders.push(f); });
  assert.deepEqual(offenders, []);
  // pure modules do not touch the page or the address at all
  ['util.js', 'content.js', 'report-view.js', 'flow.js', 'views.js', 'api-client.js', 'predictions.js', 'models.js'].forEach(f => assert.ok(!/\bdocument\b|\blocation\b|\bwindow\.|navigator\./.test(S.readWeb(f).replace(/\/\*[\s\S]*?\*\//g, '')), `${f} is pure`));
  // the only write to the address is the route hash
  const app = S.readWeb('app.js');
  assert.deepEqual([...app.matchAll(/location\.(\w+)\s*=[^=]/g)].map(m => m[1]), ['hash']);
  assert.ok(!/[?&]token|capability=|cap=/.test(app + S.readWeb('flow.js') + S.readWeb('api-client.js').replace(/\/\/.*$/gm, '')));
});

test('no local paths, user names, promotional superlatives or assistant credits in the web files', () => {
  const offenders = [];
  TEXT_FILES.forEach(f => {
    const src = S.readWeb(f);
    if (/[A-Za-z]:\\|\/Users\/|\/home\//i.test(src) || new RegExp(['s', 'h', 'e', 'r', 'o'].join(''), 'i').test(src)) offenders.push(`${f}: path or user name`);
    if (/revolutionary|groundbreaking|world-class|state-of-the-art|cutting-edge|best-in-class|unparalleled|game-changing|seamless|blazing|supercharge|magical/i.test(src)) offenders.push(`${f}: superlative`);
    if (/claude|anthropic|openai|chatgpt|gpt-|generated with|ai-generated|co-authored/i.test(src)) offenders.push(`${f}: assistant or tool credit`);
    if (/data sets?\b/i.test(src.replace(/dataset/gi, ''))) offenders.push(`${f}: use "datasets"`);
  });
  assert.deepEqual(offenders, []);
  assert.ok(/Sherwan Ali/.test(html) && S.webFiles().filter(f => f.endsWith('.js') || f.endsWith('.css')).every(f => /Sherwan Ali/.test(S.readWeb(f))), 'author wording');
});

/* ---------- landmarks, tabs and wiring ---------- */
test('the page has a language, a skip link, landmarks, one live region and a correct tab pattern', () => {
  assert.match(html, /<html lang="en">/);
  assert.match(html, /<meta name="viewport" content="width=device-width,initial-scale=1">/);
  assert.match(html, /<a class="skip-link" href="#main">Skip to main content<\/a>/);
  assert.match(html, /<header\b/); assert.match(html, /<nav class="main-nav" aria-label="Main">/); assert.match(html, /<main id="main" tabindex="-1">/); assert.match(html, /<footer>/);
  assert.equal((html.match(/id="live-status"/g) || []).length, 1);
  assert.match(html, /id="live-status" class="visually-hidden" role="status" aria-live="polite" aria-atomic="true"/);
  const tabs = [...html.matchAll(/<button type="button" role="tab" id="tab-(\w+)" aria-selected="(true|false)" aria-controls="panel-(\w+)" data-tab="(\w+)"( tabindex="-1")?>([^<]+)<\/button>/g)];
  assert.deepEqual(tabs.map(t => t[6]), ['Overview', 'Analyze a dataset', 'Evaluate predictions', 'Research', 'How it works']);
  tabs.forEach(t => { assert.equal(t[1], t[3]); assert.equal(t[1], t[4]); });
  assert.equal(tabs.filter(t => t[2] === 'true').length, 1);
  assert.equal(tabs.filter(t => !t[5]).length, 1, 'roving tabindex: only the selected tab is in the tab order');
  for (const t of tabs) assert.match(html, new RegExp(`<section id="panel-${t[1]}" role="tabpanel" aria-labelledby="tab-${t[1]}"`));
  assert.equal((html.match(/role="tabpanel"/g) || []).length, 5);
  assert.equal((html.match(/role="tabpanel"[^>]*hidden/g) || []).length, 4);
  const app = S.readWeb('app.js');
  ['ArrowLeft', 'ArrowRight', 'Home', 'End'].forEach(k => assert.ok(app.includes(`'${k}'`), k));
  assert.match(app, /tabIndex = selected \? 0 : -1/);
});

test('every form control in the static markup has an associated label', () => {
  const ids = [...html.matchAll(/<(input|select)\b[^>]*\bid="([^"]+)"/g)].map(m => m[2]);
  ids.forEach(id => assert.ok(new RegExp(`<label for="${id}"`).test(html) || /^(score-)/.test(id), `label for ${id}`));
  assert.match(html, /<fieldset><legend>Score columns<\/legend>/);
});

test('scripts load in dependency order and every web script is used by the page', () => {
  const order = [...html.matchAll(/<script defer src="([^"]+)"><\/script>/g)].map(m => m[1]);
  assert.deepEqual(order, ['preflight.js', 'scoring-core.js', 'demo-data.js', 'util.js', 'content.js', 'models.js', 'report-view.js', 'predictions.js', 'api-client.js', 'flow.js', 'views.js', 'recorded-example.js', 'tools.js', 'app.js']);
  jsFiles.forEach(f => assert.ok(order.includes(f), f + ' is loaded'));
  order.forEach(f => assert.ok(fs.existsSync(path.join(S.WEB, f)), f));
});

test('every data-action in the templates has a handler, and every id the glue looks up exists', () => {
  const app = S.readWeb('app.js');
  const sources = ['views.js', 'report-view.js', 'app.js'].map(S.readWeb).join('\n');
  const actions = new Set([...sources.matchAll(/data-action="([a-z-]+)"/g)].map(m => m[1]));
  // buttons built through the button() helper pass the action as an argument
  for (const m of S.readWeb('views.js').matchAll(/button\('([a-z-]+)'/g)) actions.add(m[1]);
  assert.ok(actions.size >= 25);
  const missing = [...actions].filter(a => !new RegExp(`['"]${a}['"]`).test(app));
  assert.deepEqual(missing, [], 'actions without a handler in app.js');
  const lookups = new Set([...app.matchAll(/\$\('([a-z0-9-]+)'\)/g)].map(m => m[1]));
  const known = html + '\n' + sources;
  const unknown = [...lookups].filter(id => !new RegExp(`id="${id}"|id=\\\\?"${id}`).test(known) && !new RegExp(`['"\`]${id}['"\`]`).test(S.readWeb('views.js')) && !['tab-', 'panel-'].includes(id));
  assert.deepEqual(unknown, []);
  // dynamic lookups
  ['tab-', 'panel-'].forEach(prefix => assert.ok(app.includes(`'${prefix}'`)));
});

test('the scoring script keeps its new name everywhere', () => {
  assert.ok(fs.existsSync(path.join(S.WEB, 'scoring-core.js')) && !fs.existsSync(path.join(S.WEB, ['metrics', 'js'].join('.'))));
  const old = ['metrics', 'js'].join('.');
  const offenders = [];
  const roots = [S.WEB, path.join(S.REPO, 'evaluation', 'tests'), path.join(S.REPO, 'evaluation', 'docs'), path.join(S.REPO, 'docs')];
  roots.forEach(dir => fs.readdirSync(dir).filter(f => /\.(js|py|html|md|css|toml|in)$/.test(f)).forEach(f => { if (fs.readFileSync(path.join(dir, f), 'utf8').includes(old)) offenders.push(path.join(path.basename(dir), f)); }));
  ['README.md', 'evaluation/README.md', 'evaluation/MANIFEST.in', 'evaluation/pyproject.toml'].forEach(f => { if (fs.readFileSync(path.join(S.REPO, f), 'utf8').includes(old)) offenders.push(f); });
  assert.deepEqual(offenders, []);
});

test('the document title changes on every route, including the example', () => {
  const app = S.readWeb('app.js');
  ['overview', 'example', 'analyze', 'evaluate', 'research', 'how'].forEach(r => assert.match(app, new RegExp(`${r}: '[A-Z][^']+'`), r));
  assert.match(app, /document\.title = `\$\{TITLES\[route\] \|\| TITLES\[tab\]\} \| FixYourDataset`/);
  const titles = Object.fromEntries([...app.slice(app.indexOf('const TITLES'), app.indexOf('};', app.indexOf('const TITLES'))).matchAll(/(\w+): '([^']+)'/g)].map(m => [m[1], m[2]]));
  assert.equal(new Set(Object.values(titles)).size, Object.keys(titles).length, 'every route has its own title');
  assert.equal(titles.example, 'Explore an example');
});

test('focus targets exist in the markup, and the live region is refilled after a short delay', () => {
  const views = S.readWeb('views.js'), app = S.readWeb('app.js'), flow = S.readWeb('flow.js');
  [...flow.matchAll(/focusOn\('([a-z-]+)'\)/g)].map(m => m[1]).forEach(id => assert.match(views, new RegExp(`id=\\\\?"${id}\\\\?"`), id));
  assert.match(views, /id="run-title" tabindex="-1"/); assert.match(views, /id="ended-title" tabindex="-1"/);
  assert.match(app, /el\.textContent = '';/); assert.match(app, /setTimeout\(\(\) => \{ el\.textContent = text;/);
  assert.match(app, /st\.focus\.n !== lastFocusN/);
});

test('literal ids written by index.html, the views and the page glue never overlap', () => {
  const literal = file => new Set([...S.readWeb(file).matchAll(/\bid="([A-Za-z0-9_-]+)"/g)].map(m => m[1]));
  const sets = { 'index.html': literal('index.html'), 'views.js': literal('views.js'), 'app.js': literal('app.js'), 'report-view.js': literal('report-view.js'), 'preflight.js': literal('preflight.js') };
  const names = Object.keys(sets), clashes = [];
  for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) sets[names[i]].forEach(id => { if (sets[names[j]].has(id)) clashes.push(`${id}: ${names[i]} and ${names[j]}`); });
  assert.deepEqual(clashes, []);
  assert.ok(sets['index.html'].has('cost-fp'));
  // Hosted controls are produced by field()/input(), not literal id attributes.
  // test_web_rendering also checks the complete rendered document at every step.
  assert.match(S.readWeb('views.js'), /input\(state, 'analyze-cost-fp'/);
  // within index.html itself
  const all = [...S.readWeb('index.html').matchAll(/\sid="([^"]+)"/g)].map(m => m[1]);
  assert.equal(new Set(all).size, all.length);
});

test('every test file named in a comment of the web folder exists', () => {
  const missing = [];
  TEXT_FILES.forEach(f => { for (const m of S.readWeb(f).matchAll(/tests\/(test_[a-z_]+\.(?:js|py))/g)) { if (!fs.existsSync(path.join(S.REPO, 'evaluation', 'tests', m[1]))) missing.push(`${f}: ${m[1]}`); } });
  assert.deepEqual(missing, []);
  assert.ok(S.readWeb('style.css').includes('tests/test_web_static.js') && S.readWeb('models.js').includes('tests/test_web_node.py'));
});

test('the analysis flow markup uses ids the error summary can link to', () => {
  const views = S.readWeb('views.js');
  ['file-input', 'target-column', 'positive-label', 'negative-label', 'group-column', 'time-column', 'seed', 'confirm-target', 'model-list', 'calibration', 'review-budgets', 'cost-fp', 'cost-fn', 'cost-review', 'start-run', 'split-mode'].forEach(id => assert.ok(views.includes(id), id));
  assert.match(views, /role="alert" tabindex="-1"/);
});

S.run('test_web_static.js');
