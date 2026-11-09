/* Sherwan Ali. Browser arithmetic checked against Python-generated fixtures. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const metrics=require('../src/rare_event_eval/web/metrics.js');
const cases=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/browser-parity.json'),'utf8'));
let compared=0;
function close(actual,expected,location){
  if(expected===null){assert.equal(actual,null,location);return;}
  if(typeof expected==='number'){assert.equal(typeof actual,'number',location);assert.ok(Math.abs(actual-expected)<=1e-11*Math.max(1,Math.abs(expected)),`${location}: ${actual} vs ${expected}`);compared++;return;}
  if(Array.isArray(expected)){assert.equal(actual.length,expected.length,location);expected.forEach((value,index)=>close(actual[index],value,location+'/'+index));return;}
  if(typeof expected==='object'){Object.entries(expected).forEach(([key,value])=>{if(key!=='interpretation')close(actual[key],value,location+'/'+key);});return;}
  assert.equal(actual,expected,location);
}
for(const [index,fixture] of cases.entries()){
  const report=metrics.evaluate(fixture.labels,fixture.scores,fixture.options);
  for(const [name,expected] of Object.entries(fixture.expected.models)){
    close(report.models[name].metrics,expected.metrics,`${index}/${name}/metrics`);
    close(report.models[name].threshold_result,expected.threshold_result,`${index}/${name}/threshold`);
    close(report.models[name].review_budget_results,expected.review_budget_results,`${index}/${name}/budgets`);
    if(expected.calibration)close(report.models[name].calibration.bins,expected.calibration.bins,`${index}/${name}/bins`);
  }
}
const parsed=metrics.parseCSV('\uFEFFlabel,score\r\n"event, rare",.9\r\n"clear ""quoted""",.1\r\n');
assert.deepEqual(parsed.rows,[['event, rare','.9'],['clear "quoted"','.1']]);
assert.throws(()=>metrics.parseCSV('label,score\nevent,.9\n\nclear,.1\n'));
assert.throws(()=>metrics.parseCSV('label,score\n"unclosed,.9\n'));
assert.throws(()=>metrics.evaluate(['event','clear'],{model:[NaN,.2]},{positiveLabel:'event',negativeLabel:'clear',scoreKind:'probability'}));
console.log(JSON.stringify({fixtures:cases.length,numerical_values_compared:compared,status:'passed'}));
