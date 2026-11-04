// Checks the honesty-receipt text of receiptText() in index.html without a browser, and that the
// page carries the exploratory note for the threshold slider.
//
// Run from the repository root (Node 18 or newer):
//     node frontend/test_receipt.js
// Exit code 0 when every check passes, 1 otherwise.
const fs = require('fs');
const path = require('path');

const file = process.argv[2] || path.join(__dirname, 'index.html');
const html = fs.readFileSync(file, 'utf8');
const a = html.indexOf('function receiptText(r)');
const b = html.indexOf('function renderDl(r)');
if (a < 0 || b < 0 || b < a) { console.error('could not locate receiptText() in ' + file); process.exit(1); }
const receiptText = eval('(' + html.slice(a, b).trim() + ')');

const base = {
  target: 'y', ir: 4, full_counts: { '0': 80, '1': 20 },
  classify: { plain: { recall: .3, pr_auc: .3, roc_auc: .7 }, csrboost: { recall: .5, pr_auc: .4, roc_auc: .8 },
              erf: { recall: .6, pr_auc: .5, roc_auc: .85 }, folds: 1 },
};
let failed = 0;
function check(label, ok) { if (!ok) failed++; console.log((ok ? 'PASS' : 'FAIL') + '  ' + label); }

const single = receiptText(base);
check('single split: says the 25% test rows are never touched and scored', /Never touched:.*25% test rows/.test(single));
check('single split: header says untouched test rows', single.includes('RESULT ON THE UNTOUCHED TEST ROWS'));

const fold = receiptText(Object.assign({}, base, { classify: Object.assign({}, base.classify, { folds: 20 }) }));
check('fold run: does not claim scores are measured on the 25% rows', !/Never touched:/.test(fold));
check('fold run: says scores are out-of-fold and the 25% split was not used',
      fold.includes('pooled out-of-fold predictions') && fold.includes('NOT used for scoring'));
check('fold run: header says out-of-fold', fold.includes('RESULT ON HELD-OUT ROWS (OUT-OF-FOLD)'));
check('fold run: keeps the fold count', fold.includes('20-fold'));
for (const [name, t] of [['single', single], ['fold', fold]]) {
  check(name + ': states the capacity difference', /1,000 trees/.test(t));
  check(name + ': no private or thesis-method claim', !/private|my thesis|mine\)|proprietary/i.test(t));
  check(name + ': no long dashes', !/[\u2013\u2014]|--/.test(t.replace(/-{10,}/g, '')));
}
check('page labels the slider as exploratory', /<b>Exploratory\.<\/b>/.test(html));

console.log(failed ? `\n${failed} check(s) FAILED` : '\nall checks passed');
process.exit(failed ? 1 : 0);
