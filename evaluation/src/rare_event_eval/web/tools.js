/* Sherwan Ali. Integrated data tools. Network calls stay in api-client.js.
   Row files are opt-in, stay in memory and are removed when inputs change. */
(function (root) {
  'use strict';
  const U = root.FYD.util, esc = U.esc;
  const TYPES = { resample: 'Resample training data', leakage: 'Leakage Lab', score: 'Score new rows' };
  const METHODS = { none: 'No resampling', random_over: 'Random oversampling', random_under: 'Random undersampling', smote: 'SMOTE / SMOTENC', clustered_smote: 'Clustered retention + SMOTE / SMOTENC' };
  function render(hosted) {
    return `<h2 tabindex="-1">Data tools</h2><p class="section-intro">Prepare training exports, examine a controlled leakage mistake, or fit a model to score additional rows. These use the same bounded service and shared package as the new analysis.</p>
      <aside class="privacy"><h3>Privacy and downloads</h3><p>Files stay in this browser until you run a tool. Running sends the selected tables to this server for temporary processing. Row-level files are returned only if you explicitly request them; they can contain sensitive data. Nothing is saved in browser storage.</p></aside>
      ${!hosted ? '<p class="notice" role="note">These tools require the hosted service. Browser-only prediction evaluation remains available without uploading data.</p>' : ''}
      <form id="tools-form" class="card" novalidate>
      <div class="field"><label for="tools-operation">Operation</label><select id="tools-operation">${Object.entries(TYPES).map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}</select></div>
      <div class="field"><label for="tools-training">Labelled training CSV (UTF-8, up to 2 MB; 2,000 rows and 50 predictors)</label><input id="tools-training" type="file" accept=".csv,text/csv"></div>
      <p id="tools-file-note" role="status">Choose a table, then confirm the labels and predictors.</p>
      <div class="field"><label for="tools-target">Target column</label><select id="tools-target"><option value="">Load a table first</option></select></div>
      <div class="field"><label for="tools-positive">Exact positive label</label><input id="tools-positive" type="text"></div>
      <div class="field"><label for="tools-negative">Exact negative label</label><input id="tools-negative" type="text"></div>
      <div class="field"><label for="tools-exclude">Exclude identifier, future-information or other metadata columns (hold Ctrl or Command for multiple)</label><select id="tools-exclude" multiple></select></div>
      <div class="field"><label for="tools-method">Resampling method (exports and Leakage Lab)</label><select id="tools-method">${Object.entries(METHODS).map(([k,v])=>`<option value="${k}"${k==='smote'?' selected':''}>${v}</option>`).join('')}</select></div>
      <div class="field"><label for="tools-model">Scoring model</label><select id="tools-model"><option value="rf_raw">Unweighted random forest, 100 trees</option><option value="logistic">Logistic regression</option><option value="erf">ERF, 20 forests of 50 trees</option></select></div>
      <div class="field"><label for="tools-scoring">Additional rows to score (UTF-8 CSV, required only for Score new rows)</label><input id="tools-scoring" type="file" accept=".csv,text/csv"></div>
      <div class="field"><label for="tools-seed">Random seed</label><input id="tools-seed" type="text" inputmode="numeric" value="42"></div>
      <div class="check-row"><input id="tools-confirm" type="checkbox"><label for="tools-confirm">I confirm the exact labels and that selected predictors are available at prediction time.</label></div>
      <div class="check-row"><input id="tools-rows" type="checkbox"><label for="tools-rows">Also return row-level downloads for my own use. They may contain sensitive data.</label></div>
      <p class="hint">Resampling uses approximately 60/20/20 training/validation/test partitions. Only training rows fit preprocessing or the honest sampler. Exports identify synthetic rows and duplicate source-pattern candidates. Scoring refits on all labelled rows and reports no new evaluation metrics.</p>
      <p class="hint">The Leakage Lab deliberately lets one sampler see validation/test labels and features. Both arms are evaluated on the identical original test rows. This is a teaching intervention, not an estimate of leakage in your real pipeline; either arm may score higher.</p>
      <div class="actions"><button id="tools-run" type="submit" class="btn btn-primary"${hosted?'':' disabled'}>Run selected tool</button><button id="tools-cancel" type="button" class="btn" disabled>Cancel</button></div>
      <p id="tools-status" role="status"></p></form><div id="tools-result"></div>`;
  }
  function resultView(data) {
    const stats = data.operation==='score' ? `<p>${esc(data.rows)} new rows scored; ${esc(data.flagged)} flagged at ${esc(data.threshold)}. ${esc(data.fit_scope)}</p>` :
      `<p>Partitions: ${esc(data.partition_counts.train)} train, ${esc(data.partition_counts.validation)} validation, ${esc(data.partition_counts.test)} test.</p><p>Training class counts (negative, positive): ${esc(data.train_counts_before.join(', '))} before, ${esc(data.train_counts_after.join(', '))} after resampling.</p>`;
    let report='';
    if(data.report) report=root.FYD.reportView.renderPredictionReport(data.report,{provenance:'fresh_app_run',idPrefix:'tools-lab'})+`<p>${esc(data.intervention)}</p>`;
    const files=Object.keys(data.downloads || {});
    return `<section class="card"><h3 tabindex="-1">${esc(TYPES[data.operation])}: result</h3><p>Positive: ${esc(data.positive_label)}. Negative: ${esc(data.negative_label)}.</p>${stats}${report}<ul>${(data.warnings||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>
      ${files.length?`<h4>Requested row-level files</h4><p>These downloads contain rows or row mappings. Keep them private.</p><div class="actions">${files.map((f,i)=>`<button type="button" class="btn" data-tools-download="${i}">Download ${esc(f)}</button>`).join('')}</div>`:'<p>No row-level files were requested or returned. Select the explicit download option before running to create them.</p>'}</section>`;
  }
  function attach({api, save, announce}) {
    let training='', scoring='', headers=[], result=null, controller=null, generation=0;
    const $=id=>root.document.getElementById(id);
    const clear=()=>{ result=null; const el=$('tools-result');if(el)el.innerHTML=''; };
    const status=text=>{if($('tools-status'))$('tools-status').textContent=U.markBidi(text);if(announce)announce(U.markBidi(text));};
    const busy=value=>{if($('tools-run'))$('tools-run').disabled=value;if($('tools-cancel'))$('tools-cancel').disabled=!value;};
    const columns=()=>{if($('tools-exclude'))$('tools-exclude').innerHTML=headers.filter(h=>h!==$('tools-target').value).map(h=>`<option value="${esc(h)}">${esc(h)}</option>`).join('');};
    root.document.addEventListener('change',async e=>{
      if(!e.target.id || !e.target.id.startsWith('tools-'))return;
      generation++; clear(); if(controller){controller.abort();controller=null;busy(false);}
      if(['tools-target','tools-positive','tools-negative','tools-exclude'].includes(e.target.id))$('tools-confirm').checked=false;
      if(e.target.id==='tools-target'){columns();return;}
      if(!['tools-training','tools-scoring'].includes(e.target.id))return;
      const file=e.target.files[0];if(!file)return;
      const inputGeneration=generation;
      try {
        if(file.size>2_000_000)throw Error('Choose a CSV up to 2 MB.');
        const text=await file.text();if(inputGeneration!==generation)return;
        const parsed=root.RareEventMetrics.parseCSV(text);
        if(e.target.id==='tools-training'){
          training=text;headers=parsed.headers;
          $('tools-target').innerHTML=headers.map(h=>`<option value="${esc(h)}">${esc(h)}</option>`).join('');
          $('tools-target').value=headers[headers.length-1];columns();$('tools-confirm').checked=false;
          $('tools-file-note').textContent=`${parsed.rows.length} rows loaded. Confirm the target and exact labels. No upload has occurred.`;
        }else scoring=text;
      }catch(err){if(inputGeneration!==generation)return;if(e.target.id==='tools-training')training='';else scoring='';status(err.message);}
    });
    root.document.addEventListener('submit',async e=>{
      if(e.target.id!=='tools-form')return;e.preventDefault();if(controller)return;
      if(!training){status('Load a labelled training CSV first.');return;}
      if(!$('tools-confirm').checked){status('Confirm the target, labels and predictors first.');return;}
      const operation=$('tools-operation').value;
      if(operation==='score'&&!scoring){status('Load the additional rows to score first.');return;}
      const seed=Number($('tools-seed').value);
      if(!Number.isInteger(seed)||seed<0||seed>=4294967296){status('Enter a valid nonnegative 32-bit integer seed.');return;}
      clear(); const ownGeneration=++generation;controller=new AbortController();busy(true);
      status('Submitted to the bounded service. It may be waiting for the shared worker or running. No progress percentage is inferred.');
      try {
        const body={csv:training,score_csv:operation==='score'?scoring:undefined,target:$('tools-target').value,positive_label:$('tools-positive').value,negative_label:$('tools-negative').value,
          exclude_columns:[...$('tools-exclude').selectedOptions].map(x=>x.value),method:$('tools-method').value,model:$('tools-model').value,seed,confirm_labels:true,include_row_data:$('tools-rows').checked};
        const answer=await api.dataTool(operation,body,controller.signal);
        if(ownGeneration!==generation)return;result=answer;$('tools-result').innerHTML=resultView(answer);
        status('Finished. Review the method, warnings and any requested downloads.');
        const heading=$('tools-result').querySelector('h3');if(heading)heading.focus();
      }catch(err){if(ownGeneration===generation)status(err.message);}
      finally{if(ownGeneration===generation){controller=null;busy(false);}}
    });
    root.document.addEventListener('click',e=>{
      if(e.target.id==='tools-cancel'){
        generation++;if(controller)controller.abort();controller=null;clear();busy(false);
        status('Cancelled in this browser. The service cancels and reaps its worker when it observes this disconnect.');
      }
      const button=e.target.closest && e.target.closest('[data-tools-download]');
      if(button&&result){const names=Object.keys(result.downloads||{}),name=names[Number(button.dataset.toolsDownload)];if(name)save(result.downloads[name],name,'text/csv;charset=utf-8');}
    });
    return {reset(){generation++;if(controller)controller.abort();controller=null;training='';scoring='';headers=[];result=null;}};
  }
  const api={render,resultView,attach,TYPES,METHODS};root.FYD.tools=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
