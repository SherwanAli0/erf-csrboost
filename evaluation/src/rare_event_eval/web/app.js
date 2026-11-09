/* Sherwan Ali. Static evaluation and an optional loopback-only training backend. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id), M = window.RareEventMetrics;
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = (v, digits=3) => v == null ? 'Unavailable' : Number(v).toLocaleString(undefined,{maximumFractionDigits:digits});
  const percent = v => v == null ? 'Unavailable' : `${number(v*100,1)}%`;
  let backend = false, predictionCSV = '', trainingCSV = '', trainingDescription = '';
  const statusText = value => ({not_checked:'Not checked',not_estimated:'Not estimated',row_indices_disjoint:'Training and test rows are disjoint'}[value] || (String(value).startsWith('not_certified:')?'Data availability and domain semantics still need review':value));
  function setTab(name) {
    document.querySelectorAll('[data-tab]').forEach(button => {
      const selected = button.dataset.tab === name;
      button.setAttribute('aria-selected',String(selected)); button.tabIndex = selected ? 0 : -1;
      $(`panel-${button.dataset.tab}`).hidden = !selected;
    });
  }
  document.querySelectorAll('[data-tab]').forEach(button => {
    button.addEventListener('click',() => setTab(button.dataset.tab));
    button.addEventListener('keydown',event => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const tabs = [...document.querySelectorAll('[data-tab]')], index=tabs.indexOf(button);
      const next = event.key==='Home' ? 0 : event.key==='End' ? tabs.length-1 : (index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
      setTab(tabs[next].dataset.tab);tabs[next].focus();
    });
  });
  function download(content,name,type) {
    const url=URL.createObjectURL(new Blob([content],{type})), link=document.createElement('a');
    link.href=url;link.download=name;link.click();setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  function reportHTML(report,view='budget',downloads=true) {
    const data=report.data, entries=Object.entries(report.models);
    const bars=entries.map(([name,row]) => `<div class="bar-row"><span>${escape(name)}</span><svg role="img" aria-label="${escape(name)} average precision ${number(row.metrics.average_precision)}" viewBox="0 0 300 12" preserveAspectRatio="none"><rect class="bar-track" width="300" height="12" rx="3"></rect><rect class="bar-fill" width="${Math.max(0,Math.min(300,(row.metrics.average_precision||0)*300))}" height="12" rx="3"></rect></svg><strong>${number(row.metrics.average_precision)}</strong></div>`).join('');
    const rows=entries.map(([name,row]) => {
      const decision=view==='threshold' ? row.threshold_result : row.review_budget_results[0];
      if (!decision) return `<tr><td>${escape(name)}</td><td colspan="7">No decision policy supplied.</td></tr>`;
      return `<tr><td>${escape(name)}</td><td>${view==='threshold' ? `Score &gt;= ${number(decision.threshold)}` : `Top ${decision.budget}`}</td><td>${decision.flagged}</td><td>${decision.true_positives}</td><td>${decision.false_positives}</td><td>${decision.false_negatives}</td><td>${percent(decision.precision)}</td><td>${percent(decision.recall)}</td></tr>`;
    }).join('');
    const metrics=entries.map(([name,row]) => `<tr><td>${escape(name)}</td><td>${number(row.metrics.average_precision)}</td><td>${number(row.metrics.roc_auc)}</td><td>${number(row.metrics.brier_score)}</td><td>${number(row.metrics.log_loss)}</td><td>${number(row.threshold_result?.assumed_total_cost,2)}</td></tr>`).join('');
    const notes=[];
    entries.forEach(([name,row]) => {
      (row.warnings||[]).forEach(note => notes.push(`${name}: ${note}`));
      row.review_budget_results.forEach(result => { if (result.boundary_tie) { const tie=result.boundary_tie;notes.push(`${name}, top ${result.budget}: tied scores can yield ${tie.true_positives_min} to ${tie.true_positives_max} positives found, depending on which tied cases are selected. These are allocation bounds, not confidence intervals.`); } });
    });
    const provenance=report.task.evaluation_role==='test' && report.execution_mode==='fresh_training'
      ? 'Fresh fit with separate training, validation and test partitions. Thresholds were chosen using validation labels only.'
      : 'Saved predictions. Class orientation and training provenance are supplied by the user, not independently verified.';
    let execution='';
    if (report.execution) {
      execution=`<div class="result-card"><h3>Execution record</h3><div class="table-wrap"><table><thead><tr><th>Model</th><th>Status</th><th>Fit seconds</th><th>Process peak MB</th></tr></thead><tbody>${Object.entries(report.execution).map(([name,e]) => `<tr><td>${escape(name)}</td><td>${escape(e.status)}</td><td>${number(e.fit_seconds,2)}</td><td>${e.peak_process_rss_bytes==null?'Unavailable':number(e.peak_process_rss_bytes/1048576,1)}</td></tr>`).join('')}</tbody></table></div><p class="table-note">Process memory includes the interpreter and libraries. Workers use one computational thread.</p></div>`;
    }
    return `<div class="stats"><div class="stat"><span>Evaluated cases</span><strong>${data.rows}</strong></div><div class="stat"><span>Positive cases</span><strong>${data.positive_rows}</strong></div><div class="stat"><span>Positive prevalence</span><strong>${percent(data.positive_prevalence)}</strong></div></div>
      <div class="result-card"><h3>Ranking quality</h3><div class="bars">${bars}</div><p class="table-note">Average precision, on the same cases. This is not a significance test or an automatic model recommendation.</p></div>
      <div class="result-card"><h3>${view==='threshold'?'Threshold decisions':'What fits the review budget?'}</h3><div class="table-wrap"><table><thead><tr><th>Model</th><th>Policy</th><th>Flagged</th><th>Found</th><th>False alarms</th><th>Missed</th><th>Precision</th><th>Recall</th></tr></thead><tbody>${rows}</tbody></table></div><p class="table-note">Positive: <strong>${escape(report.task.positive_label)}</strong>. Negative: <strong>${escape(report.task.negative_label)}</strong>. Top-k score ties use original row order.</p></div>
      <div class="result-card"><h3>Additional metrics and cost assumptions</h3><div class="table-wrap"><table><thead><tr><th>Model</th><th>AP</th><th>ROC AUC</th><th>Brier</th><th>Log loss</th><th>Threshold cost</th></tr></thead><tbody>${metrics}</tbody></table></div><p class="table-note">Cost is shown only when explicit assumptions are supplied. Brier score and log loss are unavailable for ranking scores. Reliability bins are included in the JSON.</p></div>
      ${notes.length?`<div class="result-card warnings"><h3>Read before deciding</h3><ul>${[...new Set(notes)].map(note=>`<li>${escape(note)}</li>`).join('')}</ul></div>`:''}
      <div class="result-card"><h3>What the report establishes</h3><p class="table-note">${escape(provenance)}</p><div class="check-grid"><div><strong>Label and score checks</strong>Performed</div><div><strong>Training leakage</strong><span class="not-checked">${escape(statusText(report.checks.training_leakage))}</span></div><div><strong>Row / split overlap</strong>${escape(statusText(report.checks.train_test_overlap))}</div><div><strong>Uncertainty</strong>${escape(statusText(report.checks.uncertainty))}</div></div><p class="table-note">${escape(report.data.description||'')} ${escape(report.policies.selection)}</p></div>${execution}
      ${downloads?'<div class="button-row"><button type="button" class="download" data-download="json">Download JSON</button><button type="button" class="download" data-download="html">Download report</button></div>':''}`;
  }
  function render(report,container,view='budget') {
    container.innerHTML=reportHTML(report,view);
    container.querySelector('[data-download=json]').onclick=() => download(JSON.stringify(report,null,2),'rare-event-report.json','application/json');
    container.querySelector('[data-download=html]').onclick=() => {
      const style='body{font-family:system-ui,sans-serif;max-width:1050px;margin:32px auto;padding:20px;color:#1d302d}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}.result-card{border:1px solid #ddd;border-radius:8px;padding:20px;margin:18px 0}.stats,.check-grid{display:flex;gap:20px;flex-wrap:wrap}.stat{padding:10px}.stat span{display:block}.stat strong{font-size:1.8rem}.bar-row{display:grid;grid-template-columns:150px 1fr 50px;gap:12px;margin:10px 0}.bar-track{fill:#eee}.bar-fill{fill:#245c4c}.table-wrap{overflow:auto}.table-note{font-size:.85rem;color:#62716a}';
      download(`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Evaluation report</title><style>${style}</style></head><body><h1>Rare-event evaluation</h1><p>Sherwan Ali | ${escape(report.tool.name)} ${escape(report.tool.version)}</p>${reportHTML(report,view,false)}<h2>Limitations</h2><ul>${report.limitations.map(note=>`<li>${escape(note)}</li>`).join('')}</ul></body></html>`,'rare-event-report.html','text/html');
    };
  }
  function example() {
    const item=window.ERF_EXAMPLES.find(value=>value.id===$('example-select').value), parsed=M.fromCSV(item.csv,'outcome',['baseline','candidate']);
    $('example-description').textContent=item.description;
    $('review-budget').max=parsed.labels.length;
    if (+$('review-budget').value>parsed.labels.length) $('review-budget').value=parsed.labels.length;
    $('budget-label').textContent=$('review-budget').value;$('threshold-label').textContent=(+$('example-threshold').value).toFixed(2);
    render(M.evaluate(parsed.labels,parsed.scores,{positiveLabel:'event',negativeLabel:'clear',scoreKind:'probability',
      reviewBudgets:[+$('review-budget').value],threshold:+$('example-threshold').value,description:item.description}),$('example-results'),$('decision-view').value);
  }
  $('example-select').innerHTML=window.ERF_EXAMPLES.map(item=>`<option value="${item.id}">${escape(item.title)}</option>`).join('');
  ['example-select','review-budget','example-threshold','decision-view'].forEach(id=>$(id).addEventListener('input',example));
  example();
  function options(select,headers,blank=false) { select.innerHTML=(blank?'<option value="">None</option>':'')+headers.map(name=>`<option value="${escape(name)}">${escape(name)}</option>`).join(''); }
  function scoreChoices() {
    const headers=M.parseCSV(predictionCSV).headers;
    $('score-columns').innerHTML=headers.filter(name=>name!==$('prediction-label').value).map(name=>`<label><input type="checkbox" value="${escape(name)}">${escape(name)}</label>`).join('');
  }
  function loadPrediction(csv,fictional=false) {
    const parsed=M.parseCSV(csv);predictionCSV=csv;options($('prediction-label'),parsed.headers);scoreChoices();
    $('prediction-positive').value=fictional?'event':'';$('prediction-negative').value=fictional?'clear':'';
    if (fictional) $('score-columns').querySelectorAll('input').forEach(input=>{input.checked=true;});
    $('prediction-status').textContent=`Loaded ${parsed.rows.length} rows. Confirm labels, score columns and score direction.`;
  }
  $('prediction-label').addEventListener('change',scoreChoices);
  $('use-prediction-example').onclick=()=>loadPrediction(window.ERF_EXAMPLES[0].csv,true);
  async function readFile(input) { const file=input.files[0];if(!file)return null;if(file.size>2_000_000)throw Error('Please use a CSV under 2 MB in the workbench, or use the CLI.');return file.text(); }
  $('prediction-file').onchange=async()=>{try{const text=await readFile($('prediction-file'));if(text!==null)loadPrediction(text);}catch(error){$('prediction-status').textContent=error.message;}};
  function costs() {
    const fp=$('cost-fp').value,fn=$('cost-fn').value,review=+$('cost-review').value;
    if (!fp && !fn && review===0) return null;
    if (fp==='' || fn==='') throw Error('Supply both error costs, or leave both blank.');
    return {false_positive_cost:+fp,false_negative_cost:+fn,review_cost:review};
  }
  async function post(path,body) { const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.error||'Local request failed.');return data; }
  $('prediction-form').onsubmit=async event=>{
    event.preventDefault();$('prediction-status').textContent='Evaluating...';
    try {
      if(!predictionCSV)throw Error('Load a CSV first.');
      const columns=[...$('score-columns').querySelectorAll('input:checked')].map(input=>input.value);
      const configuration={positiveLabel:$('prediction-positive').value,negativeLabel:$('prediction-negative').value,scoreKind:$('score-kind').value,
        threshold:$('prediction-threshold').value===''?null:+$('prediction-threshold').value,reviewBudgets:[+$('prediction-budget').value],costs:costs(),description:'User-supplied predictions. Training provenance has not been verified.'};
      let report;
      if (backend) report=await post('/api/evaluate',{csv:predictionCSV,label_column:$('prediction-label').value,score_columns:columns,
        positive_label:configuration.positiveLabel,negative_label:configuration.negativeLabel,score_kind:configuration.scoreKind,threshold:configuration.threshold,
        review_budgets:configuration.reviewBudgets,costs:configuration.costs});
      else {const parsed=M.fromCSV(predictionCSV,$('prediction-label').value,columns);report=M.evaluate(parsed.labels,parsed.scores,configuration);}
      render(report,$('prediction-results'));$('prediction-status').textContent=backend?'Evaluated by the local Python core. Uploaded rows were not stored.':'Evaluated in your browser. No CSV was sent to a server.';
    }catch(error){$('prediction-status').textContent=error.message;$('prediction-results').innerHTML='<div class="empty-state"><h3>Input needs attention</h3><p>The previous result is cleared. Correct the issue shown beside the form and evaluate again.</p></div>';}
  };
  function loadTraining(csv,fictional=false) {
    const parsed=M.parseCSV(csv);trainingCSV=csv;options($('training-target'),parsed.headers);options($('training-group'),parsed.headers,true);options($('training-time'),parsed.headers,true);
    trainingDescription=fictional?'Fictional training cases generated for software demonstration; not a real-world case study.':'User-supplied numeric training data; provenance is not independently verified.';
    if(fictional){$('training-target').value='outcome';$('training-positive').value='event';$('training-negative').value='clear';}
    else {$('training-positive').value='';$('training-negative').value='';}
    $('training-file-note').textContent=`${parsed.rows.length} rows loaded. ${fictional?'Fictional teaching data.':'Confirm which columns are available at prediction time.'}`;
  }
  $('use-training-example').onclick=()=>loadTraining(window.ERF_TRAINING_CSV,true);
  $('training-file').onchange=async()=>{try{const text=await readFile($('training-file'));if(text!==null)loadTraining(text);}catch(error){$('training-status').textContent=error.message;}};
  function combineTraining(result) {
    const successes=Object.entries(result.outcomes).filter(([,row])=>row.status==='success');
    if(!successes.length)throw Error('No successful fit. '+Object.entries(result.outcomes).map(([name,row])=>`${name}: ${row.status}, ${row.error||'worker limit reached'}`).join('; '));
    const report=JSON.parse(JSON.stringify(successes[0][1].report));report.models=Object.create(null);report.execution=Object.create(null);
    successes.forEach(([name,row])=>{report.models[name]=row.report.models[name];});
    Object.entries(result.outcomes).forEach(([name,row])=>{report.execution[name]={status:row.status,fit_seconds:row.execution?.fit_seconds,peak_process_rss_bytes:row.execution?.peak_process_rss_bytes};if(row.status!=='success')report.limitations.push(`${name}: ${row.status}, ${row.error||'worker limit reached'}`);});
    report.policies.selection='Per-model thresholds chosen on validation data only. Test labels excluded from selection.';report.policies.threshold=null;return report;
  }
  $('training-form').onsubmit=async event=>{
    event.preventDefault();if(!backend)return;
    $('train-button').disabled=true;$('training-status').textContent='Running bounded workers on this machine. A model may take up to 60 seconds.';
    try {
      if(!trainingCSV)throw Error('Load a training CSV first.');
      const models=[...document.querySelectorAll('input[name=model]:checked')].map(input=>input.value);
      const result=await post('/api/train',{csv:trainingCSV,target:$('training-target').value,positive_label:$('training-positive').value,
        negative_label:$('training-negative').value,models,split_mode:$('training-split').value,group_column:$('training-group').value||null,
        time_column:$('training-time').value||null,calibration:$('training-calibration').value,data_description:trainingDescription});
      render(combineTraining(result),$('training-results'),'threshold');$('training-status').textContent='Finished. Inspect validation-selected policies, held-out results and any model failures.';
    }catch(error){$('training-status').textContent=error.message;}finally{$('train-button').disabled=!backend;}
  };
  function study() {
    const data=window.ERF_STUDY;
    if(!data){$('study-results').innerHTML='<div class="result-card reading"><h3>A new controlled pilot</h3><p>The study record is being prepared in this local development version. Historical claims are not substituted for a fresh result.</p></div>';return;}
    const rows=data.family_comparisons.map(row=>`<tr><td>${escape(row.dataset)}</td><td>${number(row.erf.mean_ap)}</td><td>${number(row.balanced_rf.mean_ap)}</td><td>${number(row.rf_weighted_1000.mean_ap)}</td><td>${number(row.hist_gradient_boosting.mean_ap)}</td><td>${number(row.erf_minus_balanced_rf_ap)}</td></tr>`).join('');
    $('study-results').innerHTML=`<div class="study-header"><div class="stat"><span>Dataset families</span><strong>5</strong></div><div class="stat"><span>Planned runs</span><strong>${data.planned_jobs}</strong></div><div class="stat"><span>Successful runs</span><strong>${data.status_counts.success||0}</strong></div></div><div class="result-card reading"><h3>Decision from the prespecified gate</h3><p>${escape(data.confirmation_gate.decision)}</p><p>Median ERF / balanced-forest fit-time ratio: <strong>${number(data.confirmation_gate.median_erf_to_balanced_rf_fit_ratio,1)}x</strong>. This is a computational comparison, not an estimate of business value.</p></div><div class="result-card"><h3>Within-family comparison</h3><div class="table-wrap"><table><thead><tr><th>Task</th><th>ERF AP</th><th>Balanced RF AP</th><th>1,000-tree RF AP</th><th>Gradient boosting AP</th><th>ERF - balanced RF</th></tr></thead><tbody>${rows}</tbody></table></div><p class="table-note">Each value is the mean of two predeclared stratified splits. Families have different prevalence; do not interpret cross-task average precision as one universal performance score. Group sensitivity results, failures and all ablations are in the local study files.</p></div><div class="result-card reading"><h3>Limits of this evidence</h3><ul>${data.limitations.map(note=>`<li>${escape(note)}</li>`).join('')}</ul><p class="mono">Study identity: ${escape(data.identity_sha256)}</p></div>`;
  }
  study();
  async function checkBackend(){
    if(location.protocol==='file:' || new URLSearchParams(location.search).get('mode')==='browser')return;
    try{const response=await fetch('/api/health',{signal:AbortSignal.timeout(2000)});if(!response.ok)return;const data=await response.json();if(data.mode!=='local')return;backend=true;$('runtime-badge').textContent='Local Python connected';$('training-mode').textContent='Training available locally';$('train-button').disabled=false;$('local-instructions').hidden=true;}catch(_){/* Static hosting intentionally has no API. */}
  }
  checkBackend();
})();
