/* Sherwan Ali. Integrated tool privacy, labels, escaping, routing and cancellation contract. */
'use strict';
const assert=require('node:assert/strict');
const S=require('./web_support.js');
const {TOOLS,API,V,FLOW}=S.load();const {test}=S;
test('tools are part of the workbench and hosted actions are unavailable in static mode',()=>{
  const page=TOOLS.render(true);
  assert.deepEqual(S.tagAudit(page,['form']),[]);
  assert.deepEqual(S.duplicateIds(S.readWeb('index.html')+page),[]);
  for(const name of ['Resample training data','Leakage Lab','Score new rows'])assert.ok(page.includes(name));
  assert.ok(TOOLS.render(false).includes('id="tools-run" type="submit" class="btn btn-primary" disabled'));
  assert.ok(!/id="tools-rows"[^>]*checked/.test(page));
  assert.ok(page.includes('Both arms are evaluated on the identical original test rows'));
});
test('sensitive downloads require opt-in and all displayed labels and file names are escaped',()=>{
  const hostile='<script>alert(1)</script>'+String.fromCharCode(0x202E);
  const html=TOOLS.resultView({operation:'score',rows:2,flagged:1,threshold:.5,positive_label:hostile,negative_label:'clear',fit_scope:hostile,warnings:[hostile],downloads:{[hostile]:'raw input rows'}});
  assert.deepEqual(S.tagAudit(html),[]);
  assert.ok(html.includes('&lt;script&gt;')&&html.includes('&lt;U+202E&gt;'));
  assert.ok(!html.includes('raw input rows'));
  assert.ok(TOOLS.resultView({operation:'score',rows:0,flagged:0,positive_label:'x',negative_label:'y',warnings:[],downloads:{}}).includes('No row-level files were requested'));
});
test('network request has an abort signal, typed JSON and no shareable row-result URL',async()=>{
  let seen;
  const client=API.createApiClient({fetchImpl:async(path,init)=>{seen={path,init};return {ok:true,status:200,headers:{get:()=>null},json:async()=>({ok:true})};}});
  const signal=new AbortController().signal;
  await client.dataTool('score',{confirm_labels:true,include_row_data:false},signal);
  assert.equal(seen.path,'/api/v1/tools/score');assert.equal(seen.init.signal,signal);
  assert.equal(JSON.parse(seen.init.body).include_row_data,false);
  assert.equal(seen.init.headers['Content-Type'],'application/json');
});
test('route, script preflight and cancellation are wired into the shared page',()=>{
  const app=S.readWeb('app.js'),tools=S.readWeb('tools.js');
  assert.match(app,/tools: 'analyze'/);assert.match(app,/F\.tools\.attach/);
  assert.match(app,/route === 'tools'/);
  assert.ok(S.readWeb('preflight.js').includes("['tools.js'"));
  assert.ok(tools.includes('controller.abort()'));
  assert.ok(!/localStorage|sessionStorage|console\.|fetch\(/.test(tools));
});
S.run('test_web_tools.js');
