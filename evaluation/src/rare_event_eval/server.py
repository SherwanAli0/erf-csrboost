"""Local-only workbench, no accounts, visitor logging or persistence of uploads."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import unquote, urlparse

from .cli import parse_prediction_csv
from .core import CostAssumptions, evaluate_predictions
from .datasets import parse_training_csv
from .tabular import excluded_metadata_records
from .training import train_comparison, MODEL_NAMES

WEB = Path(__file__).parent/'web'
ASSETS = {'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8',
          'metrics.js':'text/javascript; charset=utf-8','demo-data.js':'text/javascript; charset=utf-8',
          'style.css':'text/css; charset=utf-8'}
ASSETS['quality-vs-cost.png']='image/png'
BUSY = threading.BoundedSemaphore(1)


class Handler(BaseHTTPRequestHandler):
    timeout = 15
    def log_message(self, *_):
        pass

    def _host_ok(self):
        allowed = {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            self._send(403,{'error':'Only the local workbench origin is accepted.'})
            return False
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://{host}' for host in allowed}:
            self._send(403,{'error':'Cross-origin requests are not accepted.'})
            return False
        return True

    def _send(self,status,payload,content_type='application/json; charset=utf-8'):
        data=json.dumps(payload,allow_nan=False).encode() if not isinstance(payload,bytes) else payload
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command!='HEAD':
            self.wfile.write(data)

    def do_GET(self):
        if not self._host_ok():
            return
        path=unquote(urlparse(self.path).path)
        if path in ('/api/health','/healthz'):
            self._send(200,{'ok':True,'mode':'local','training_available':True,'uploads_stored':False})
            return
        name='index.html' if path=='/' else path.lstrip('/')
        if name not in ASSETS or not (WEB/name).is_file():
            self._send(404,{'error':'Not found.'})
            return
        self._send(200,(WEB/name).read_bytes(),ASSETS[name])

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        if not self._host_ok():
            return
        path=urlparse(self.path).path
        if path not in ('/api/evaluate','/api/train'):
            self._send(404,{'error':'Not found.'})
            return
        if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
            self._send(415,{'error':'Send application/json.'})
            return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<1 or length>2_000_000:
                self._send(413,{'error':'Local API accepts JSON requests up to 2 MB.'})
                return
            request=json.loads(self.rfile.read(length))
            if not isinstance(request,dict) or not isinstance(request.get('csv'),str):
                raise ValueError('A CSV string is required.')
            raw=request['csv'].encode('utf-8')
            costs=CostAssumptions(**request['costs']) if request.get('costs') is not None else None
            if path=='/api/evaluate':
                labels,scores,checksum=parse_prediction_csv(raw,request['label_column'],request['score_columns'])
                result=evaluate_predictions(labels,scores,positive_label=request['positive_label'],negative_label=request['negative_label'],
                    score_kind=request['score_kind'],threshold=request.get('threshold'),review_budgets=request.get('review_budgets',[]),
                    costs=costs,evaluation_role='unspecified',data_description='User-supplied predictions evaluated through the local core.')
                result['data']['input_sha256']=checksum
                self._send(200,result)
                return
            models=request.get('models',['logistic','rf_weighted'])
            description=request.get('data_description','User-supplied numeric data. Provenance is not independently verified.')
            if not isinstance(description,str) or not description.strip() or len(description)>500:
                raise ValueError('Data description must contain 1..500 characters.')
            if not isinstance(models,list) or not 1<=len(models)<=5 or any(name not in MODEL_NAMES for name in models):
                raise ValueError('Choose one to five supported models.')
            dataset=parse_training_csv(raw,target=request['target'],group_column=request.get('group_column'),time_column=request.get('time_column'))
            if not BUSY.acquire(blocking=False):
                self._send(429,{'error':'One training job is already running. Try again after it finishes.'})
                return
            try:
                result=train_comparison(dataset['X'],dataset['labels'],models=models,positive_label=request['positive_label'],
                    negative_label=request['negative_label'],split_mode=request.get('split_mode','stratified'),groups=dataset['groups'],
                    times=dataset['times'],seed=42,calibration=request.get('calibration','none'),timeout_seconds=60,costs=costs,
                    report_context={'feature_names':dataset['feature_names'],'input_sha256':dataset['input_sha256'],
                                    'parser':'parse_training_csv (numeric, strict; malformed rows raise)',
                                    'excluded_columns':excluded_metadata_records(request['target'],request.get('group_column'),request.get('time_column'))})
                result['input_sha256']=dataset['input_sha256']
                result['feature_names']=dataset['feature_names']
                for row in result['outcomes'].values():
                    if row['status']=='success':
                        row['report']['data']['description']=description+' Fresh fit with separate evaluation partitions.'
                        row['execution'].get('model_parameters',{}).pop('cv',None)
            finally:
                BUSY.release()
            self._send(200,result)
        except (ValueError,KeyError,TypeError,UnicodeError) as exc:
            self._send(400,{'error':str(exc)})


def serve(port=8601):
    if not isinstance(port,int) or not 1<=port<=65535:
        raise ValueError('Port must be from 1 to 65535.')
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Local workbench: http://127.0.0.1:{port}. Stop with Ctrl+C. Uploaded rows are not stored.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def export_demo(output_directory):
    output_directory=Path(output_directory)
    if output_directory.exists():
        raise ValueError('The output directory already exists. Choose a new directory.')
    shutil.copytree(WEB,output_directory)
