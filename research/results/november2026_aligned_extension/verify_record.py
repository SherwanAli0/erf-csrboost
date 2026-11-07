"""Verify stored tables only. Author: Sherwan Ali."""
from pathlib import Path
import csv,hashlib,json,collections,math
ROOT=Path(__file__).resolve().parent
def read(name):
    with (ROOT/name).open(newline='',encoding='utf-8') as s:return list(csv.DictReader(s))
for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
    expected,name=line.split('  ',1);assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
rows=read('fold_results.csv');assert len(rows)==2000 and all(r['status']=='success' for r in rows)
ids={(r['dataset'],r['method'],int(r['fold'])) for r in rows};assert len(ids)==2000
assert set((r['method'] for r in rows))=={'CSRBOOST','ERF_CSRBOOST','RF_RAW','RF_BRF'}
groups=collections.defaultdict(list)
for r in rows:
    assert 0<=int(r['fold'])<100
    for k in ['acc','auc','f1','ap','gmean']:assert 0<=float(r[k])<=1
    groups[r['dataset'],r['method']].append(float(r['ap']))
assert len(groups)==20 and all(len(v)==100 for v in groups.values())
combined=read('aligned_20_task_ap.csv');assert len(combined)==20
oldpath=ROOT.parent/'comparison_june2026_four_way/fold_results.csv'
protocol=json.loads((ROOT/'protocol.json').read_text())
assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==protocol['historical_reference_sha256']
with oldpath.open(newline='',encoding='utf-8') as s:historical=list(csv.DictReader(s))
old=collections.defaultdict(list)
for r in historical:old[r['dataset'],r['method']].append(float(r['ap']))
assert len(historical)==6000 and len(old)==60
for r in combined:
    source=groups if r['source']=='new_june_protocol_extension_5' else old
    for m in ['CSRBOOST','ERF_CSRBOOST','RF_RAW','RF_BRF']:
        values=source[r['dataset'],m];assert math.isclose(sum(values)/len(values),float(r[m]),rel_tol=1e-12,abs_tol=1e-12)
counts={}
for m in ['CSRBOOST','RF_RAW','RF_BRF']:
    delta=[float(r['ERF_CSRBOOST'])-float(r[m]) for r in combined]
    counts[m]={'higher':sum(d>1e-12 for d in delta),'ties':sum(abs(d)<=1e-12 for d in delta),'lower':sum(d< -1e-12 for d in delta)}
assert counts['CSRBOOST']=={'higher':16,'ties':0,'lower':4}
print(json.dumps({'verified_new_rows':len(rows),'datasets':len(combined),'AP_comparisons':counts},indent=2))
