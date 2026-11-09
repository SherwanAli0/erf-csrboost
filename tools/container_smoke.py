"""Exercise the production image at one CPU and 1 GB, without Fly or secrets.

Author: Sherwan Ali. Only seeded fictional inputs are used. The temporary
container is removed in finally; existing containers and volumes are untouched.
"""
from __future__ import annotations
import argparse,base64,csv,gzip,io,json,subprocess,threading,time,urllib.request,urllib.error,uuid


def docker(*args,check=True):
    r=subprocess.run(['docker',*args],capture_output=True,text=True)
    if check and r.returncode:raise RuntimeError('docker '+args[0]+' failed: '+r.stderr[-1200:])
    return r.stdout.strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',required=True);parser.add_argument('--output')
    args=parser.parse_args();name='fyd-smoke-'+uuid.uuid4().hex[:12]
    checks=[];stop=threading.Event();samples=[];sampler=None
    def note(label,ok=True):
        checks.append({'check':label,'passed':bool(ok)})
        if not ok:raise AssertionError(label)
        print('PASS '+label,flush=True)
    try:
        docker('run','-d','--name',name,'--memory=1g','--cpus=1','--read-only',
               '--tmpfs','/tmp:rw,size=128m','--tmpfs','/data:rw,uid=1000,gid=1000,size=8m',
               '-p','127.0.0.1::7860','-e','FYD_TEMP_ROOT=/tmp/fyd-jobs','-e','FYD_USAGE_DB=/tmp/fyd-usage.db',
               '-e','FYD_RELEASE_ID=container-check',args.image)
        port=docker('port',name,'7860/tcp').splitlines()[0].rsplit(':',1)[1];base='http://127.0.0.1:'+port
        def request(route,body=None,token=None,method=None,headers=None,timeout=75):
            h=dict(headers or {});data=None
            if token:h['Authorization']='Bearer '+token
            if body is not None:
                data=body if isinstance(body,bytes) else json.dumps(body).encode();h.setdefault('Content-Type','application/octet-stream' if isinstance(body,bytes) else 'application/json')
            q=urllib.request.Request(base+route,data=data,headers=h,method=method or ('POST' if body is not None else 'GET'))
            try:r=urllib.request.urlopen(q,timeout=timeout)
            except urllib.error.HTTPError as exc:r=exc
            raw=r.read();return r.code,json.loads(raw) if raw and 'application/json' in r.headers.get('Content-Type','') else raw
        until=time.monotonic()+90
        while True:
            try:
                if request('/healthz',timeout=2)[0]==200:break
            except OSError:pass
            if time.monotonic()>until:raise RuntimeError('Container failed readiness: '+docker('logs',name,check=False)[-1200:])
            time.sleep(.25)
        note('production entry point answers health')
        config=json.loads(docker('inspect',name))[0]
        note('one CPU and exactly 1 GB configured',config['HostConfig']['Memory']==1073741824 and config['HostConfig']['NanoCpus']==1000000000)
        def sample_memory():
            while not stop.is_set():
                try:samples.append(json.loads(docker('stats','--no-stream','--format','{{json .}}',name)))
                except (ValueError,RuntimeError):pass
                stop.wait(1)
        sampler=threading.Thread(target=sample_memory,daemon=True);sampler.start()
        for route in ('/','/healthz','/tools.js','/classic'):
            note('GET and HEAD '+route,request(route)[0]==200 and request(route,method='HEAD')[0]==200)
        note('version identifies test release',request('/api/v1/version')[1]['release_id']=='container-check')
        raw=io.StringIO();writer=csv.writer(raw,lineterminator='\n');writer.writerow(['value','region','outcome'])
        for i in range(320):writer.writerow([i/7,'north' if i%2 else 'south','event' if i%8==0 else 'clear'])
        table=raw.getvalue().encode()
        def dataset():
            status,data=request('/api/v1/datasets',table,headers={'X-Filename':'fictional.csv'})
            note('fictional dataset accepted',status==201);return data
        def job(ds,models):
            return request('/api/v1/jobs',{'dataset_id':ds['dataset_id'],'target':'outcome','positive_label':'event','negative_label':'clear','models':models,'seed':42},ds['capability'])
        def status(j,token=None):return request('/api/v1/jobs/'+j['job_id'],token=token or j['capability'])
        def done(j):
            until=time.monotonic()+120
            while time.monotonic()<until:
                code,s=status(j)
                if code==200 and s['state'] not in ('queued','running'):return s
                time.sleep(.2)
            raise AssertionError('Job did not reach a terminal state')
        ds=dataset()
        for i in range(3):
            code,j=job(ds,['logistic','rf_raw']);note('comparison accepted '+str(i+1),code==202)
            result=done(j);note('comparison succeeds '+str(i+1),result['state']=='succeeded')
            code,report=request('/api/v1/jobs/'+j['job_id']+'/result',token=j['capability'])
            note('held-out report returned '+str(i+1),code==200 and bool(report))
            for kind in ('report_json','report_html','config_json','repro_bundle'):
                note('aggregate export '+kind,request('/api/v1/jobs/'+j['job_id']+'/exports/'+kind,token=j['capability'])[0]==200)
            note('wrong capability cannot read result',status(j,'wrong-capability')[0] in (401,404))
        datasets=[ds,dataset(),dataset()];live=[]
        for d in datasets[:2]:
            code,j=job(d,['erf']);note('bounded heavy lane accepts job',code==202);live.append(j)
        code,_=job(datasets[2],['erf']);note('third heavy request is rejected promptly',code==429)
        start=time.monotonic();note('health remains responsive while fitting',request('/healthz',timeout=3)[0]==200 and time.monotonic()-start<3)
        for j in live:request('/api/v1/jobs/'+j['job_id'],token=j['capability'],method='DELETE')
        for j in live:note('cancelled job reaches terminal state',done(j)['state']=='cancelled')
        code,data=request('/api/v1/tools/resample',{'csv':table.decode(),'target':'outcome','positive_label':'event','negative_label':'clear','confirm_labels':True,'method':'smote','include_row_data':False})
        note('integrated tool works within container limit',code==200 and data['downloads']=={})
        wide=io.StringIO();w=csv.writer(wide,lineterminator='\n')
        w.writerow([f'n{k}' for k in range(20)]+[f'c{k}' for k in range(4)]+['outcome'])
        for i in range(10000):
            w.writerow([(i*(k+3)+k)%997 for k in range(20)]+[f'cat{(i//(k+1))%20}' for k in range(4)]+['event' if i%8==0 else 'clear'])
        code,large=request('/api/v1/datasets',wide.getvalue().encode(),headers={'X-Filename':'wide-fictional.csv'})
        note('10,000-row wide fictional input accepted',code==201)
        code,large_job=job(large,['logistic','erf']);note('large bounded comparison accepted',code==202)
        # Exercise the light conversion lane while a heavy fit is live.
        code,_=request('/api/convert',{'name':'fictional.csv','b64':base64.b64encode(table).decode()})
        note('light conversion remains available during heavy work',code==200)
        note('health responds during wide-input fit',request('/healthz',timeout=3)[0]==200)
        large_status=done(large_job)
        note('wide-input run terminates within its bounds',large_status['state'] in ('succeeded','partial','failed','timed_out'))
        print('Wide-input outcome: '+json.dumps({'state':large_status['state'],'models':large_status.get('models',[])}),flush=True)
        packed=io.BytesIO()
        with gzip.GzipFile(fileobj=packed,mode='wb') as f:
            chunk=b'x\n'*(1024*512)
            for _ in range(83):f.write(chunk)
        code,_=request('/api/convert',{'name':'oversized.csv.gz','b64':base64.b64encode(packed.getvalue()).decode()})
        note('oversized decompression is refused',code in (400,413,422,503))
        note('service survives refused expansion',request('/healthz',timeout=3)[0]==200)
        inspect=json.loads(docker('inspect',name))[0]
        note('container is running without OOM kill',inspect['State']['Running'] and not inspect['State']['OOMKilled'])
        stop.set();sampler.join(timeout=10)
        receipt={'image':args.image,'checks':checks,'resource_limit':{'cpus':1,'memory_bytes':1073741824},'memory_samples':samples,
                 'wide_input_status':large_status,
                 'scope':'Local constrained container, fictional tables; no Fly resources or production data used.'}
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'passed':len(checks),'memory_samples':len(samples)}))
    finally:
        stop.set()
        if sampler:sampler.join(timeout=10)
        docker('rm','-f',name,check=False)


if __name__=='__main__':main()
