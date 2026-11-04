// Checks the verdict logic of renderVerdict() and the results nav chip (chipFor) in index.html
// without a browser. The functions are cut out of the page source and run against stub inputs.
//
// Run from the repository root (Node 18 or newer):
//     node frontend/test_verdict.js
// Optional: pass another html file as the first argument.
//
// Exit code 0 when every case behaves as expected, 1 otherwise.
const fs = require('fs');
const path = require('path');

const file = process.argv[2] || path.join(__dirname, 'index.html');
const html = fs.readFileSync(file, 'utf8');
const a = html.indexOf('function renderVerdict(r)');
const b = html.indexOf('function renderLab(r)');
if (a < 0 || b < 0 || b < a) {
  console.error('could not locate renderVerdict() in ' + file);
  process.exit(1);
}

const store = {};
global.$ = () => ({
  set innerHTML(v) { store.html = v; },
  get innerHTML() { return store.html; },
});
eval(html.slice(a, b));

const plain = { recall: 0.30, pr_auc: 0.30 };
let failed = 0;

// chipText: the text the results nav chip must show. The chip must always carry the same colour
// class as the verdict box, so a tie or a loss can never show up as "ahead".
function run(label, csr, erf, expectClass, mustInclude, folds, extra, chipText, ir) {
  store.html = '';
  const r = { ir: ir || 4, classify: Object.assign({ plain, csrboost: csr, erf, folds: folds || 1 }, extra || {}) };
  renderVerdict(r);
  const chip = chipFor(r);
  const cls = (store.html.match(/class="verdict (\w+)"/) || [])[1];
  const txt = store.html.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
  const okClass = cls === expectClass;
  const okText = (mustInclude || []).every(s => txt.includes(s));
  const bad = /my thesis method|on your data|matched the published/.test(txt);
  const okChip = chip.cls === cls && chip.text === chipText;
  const ok = okClass && okText && !bad && okChip;
  if (!ok) failed++;
  console.log((ok ? 'PASS' : 'FAIL') + '  ' + label.padEnd(52) + '| ' + cls + ' | chip: ' + chip.text +
              (okChip ? '' : ' (WANTED ' + chipText + ', ' + cls + ')') + ' | ' + txt.slice(0, 90));
}

// CSRBoost first, ERF second
run('ERF much worse on recall, PR-AUC equal',
    { recall: 0.80, pr_auc: 0.50 }, { recall: 0.50, pr_auc: 0.50 }, 'amber', ['recall 80% to 50%'],
    1, null, 'CSRBoost ahead');
run('ERF much worse on recall, PR-AUC +0.01',
    { recall: 0.80, pr_auc: 0.50 }, { recall: 0.50, pr_auc: 0.51 }, 'amber', ['recall 80% to 50%'],
    1, null, 'CSRBoost ahead');
run('ERF worse on both (-0.05, -0.05)',
    { recall: 0.80, pr_auc: 0.50 }, { recall: 0.75, pr_auc: 0.45 }, 'amber', ['came out ahead of ERF-CSRBoost'],
    1, null, 'CSRBoost ahead');
run('ERF slightly better on recall (+0.02)',
    { recall: 0.70, pr_auc: 0.50 }, { recall: 0.72, pr_auc: 0.50 }, 'good', ['ERF-CSRBoost came out ahead'],
    1, null, 'ERF-CSRBoost ahead');
run('ERF recall -0.10, PR-AUC +0.03 (mixed)',
    { recall: 0.80, pr_auc: 0.50 }, { recall: 0.70, pr_auc: 0.53 }, 'amber', ['Mixed result'],
    1, null, 'Mixed result');
run('Within 0.02 on both (about the same)',
    { recall: 0.80, pr_auc: 0.50 }, { recall: 0.81, pr_auc: 0.51 }, 'good', ['about the same'],
    1, null, 'About the same');
run('Fold run keeps the fold-count sentence',
    { recall: 0.70, pr_auc: 0.50 }, { recall: 0.80, pr_auc: 0.60 }, 'good', ['7 of 20 folds'], 20, { wins_erf_vs_csr: 7 },
    'ERF-CSRBoost ahead');
// chip-only cases (the chip used to say "ERF-CSRBoost ahead" whenever the verdict was not amber)
run('Exact tie: chip must not say ahead',
    { recall: 0.80, pr_auc: 0.50 }, { recall: 0.80, pr_auc: 0.50 }, 'good', ['about the same'],
    1, null, 'About the same');
run('Gain of 0.019 is still a tie',
    { recall: 0.70, pr_auc: 0.50 }, { recall: 0.719, pr_auc: 0.50 }, 'good', ['about the same'],
    1, null, 'About the same');
run('Loss of 0.02 on PR-AUC only (behind)',
    { recall: 0.70, pr_auc: 0.50 }, { recall: 0.70, pr_auc: 0.48 }, 'amber', ['came out ahead of ERF-CSRBoost'],
    1, null, 'CSRBoost ahead');
run('Already balanced column (ratio 1.2)',
    { recall: 0.70, pr_auc: 0.50 }, { recall: 0.80, pr_auc: 0.60 }, 'amber', ['already fairly balanced'],
    1, null, 'Already balanced', 1.2);
run('No CSRBoost result, recall up 0.05 (helped)',
    null, { recall: 0.35, pr_auc: 0.50 }, 'good', ['Balancing helped'], 1, null, 'Balancing helped');
run('No CSRBoost result, recall up 0.01 (no clear lift)',
    null, { recall: 0.31, pr_auc: 0.50 }, 'amber', ['did not clearly help'], 1, null, 'No clear lift');

console.log(failed ? `\n${failed} case(s) FAILED` : '\nall cases passed');
process.exit(failed ? 1 : 0);
