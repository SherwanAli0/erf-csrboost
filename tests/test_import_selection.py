"""Explicit multi-table imports and bounded container readers. Author: Sherwan Ali."""
import gzip,io,json,unittest,zipfile
from urllib.parse import quote
import api_support
from api_support import ApiCase,auth
from services.profile import profile_for_app
from services.errors import ServiceError
from rare_event_eval.tabular import TabularLimits

CSV='value,outcome\n'+''.join(f'{i},'+('event' if i%8==0 else 'clear')+'\n' for i in range(160))

def archive(files):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,body in files:z.writestr(name,body)
    return out.getvalue()

def workbook():
    from openpyxl import Workbook
    book=Workbook();book.active.title='first';book.active.append(['value','outcome']);book.active.append([1,'event'])
    sheet=book.create_sheet('chosen');sheet.append(['value','outcome'])
    for i in range(160):sheet.append([i,'event' if i%8==0 else 'clear'])
    out=io.BytesIO();book.save(out);return out.getvalue()

class ImportContract(unittest.TestCase):
    def test_multi_member_requires_choice_and_keeps_original_hash(self):
        raw=archive([('first.csv','x,y\n1,2\n'),('folder/chosen.csv',CSV)])
        with self.assertRaises(ServiceError) as caught:profile_for_app(raw,'tables.zip')
        self.assertEqual(caught.exception.code,'APP_IMPORT_SELECTION_REQUIRED')
        self.assertEqual(caught.exception.details,{'selection_kind':'member','choices':['first.csv','folder/chosen.csv']})
        report=profile_for_app(raw,'tables.zip',import_selection={'member':'folder/chosen.csv'})
        self.assertEqual(report['rows'],160)
        self.assertEqual(report['parsing']['import_selection'],{'member':'folder/chosen.csv'})

    def test_multi_sheet_requires_choice(self):
        raw=workbook()
        with self.assertRaises(ServiceError) as caught:profile_for_app(raw,'book.xlsx')
        self.assertEqual(caught.exception.details['selection_kind'],'sheet')
        self.assertEqual(profile_for_app(raw,'book.xlsx',import_selection={'sheet':'chosen'})['rows'],160)

    def test_archive_with_workbook_preserves_both_choices(self):
        raw=archive([('book.xlsx',workbook()),('other.csv',CSV)])
        result=profile_for_app(raw,'pack.zip',import_selection={'member':'book.xlsx','sheet':'chosen'})
        self.assertEqual(result['rows'],160)
        self.assertEqual(result['parsing']['import_selection']['sheet'],'chosen')

    def test_unsafe_paths_and_expansion_are_refused(self):
        for name in ('../bad.csv','/bad.csv','C:/bad.csv','folder/../../bad.csv'):
            with self.subTest(name=name),self.assertRaises(ServiceError):profile_for_app(archive([(name,CSV)]),'pack.zip')
        with self.assertRaises(ServiceError):profile_for_app(gzip.compress(b'x'*5000),'table.csv.gz',limits=TabularLimits(max_bytes=1000))
        with self.assertRaises(ServiceError):profile_for_app(archive([('data.csv','x'*5000)]),'pack.zip',limits=TabularLimits(max_bytes=1000))

    def test_bad_selection_and_missing_table_are_refused(self):
        raw=archive([('one.csv',CSV)])
        for selection in ({'member':'missing.csv'},{'other':'one.csv'},['one.csv']):
            with self.subTest(selection=selection),self.assertRaises(ServiceError):profile_for_app(raw,'pack.zip',import_selection=selection)

class ImportHttp(ApiCase):
    def test_selection_survives_upload_and_training_worker(self):
        raw=archive([('one.csv','x,y\n1,2\n'),('chosen.csv',CSV)])
        headers={'Content-Type':'application/octet-stream','X-Filename':'pack.zip'}
        refused=self.client.post('/api/v1/datasets',content=raw,headers=headers)
        self.assertEqual(refused.status_code,422,refused.text)
        self.assertEqual(refused.json()['error']['code'],'APP_IMPORT_SELECTION_REQUIRED')
        headers['X-Import-Selection']=quote(json.dumps({'member':'chosen.csv'}))
        created=self.client.post('/api/v1/datasets',content=raw,headers=headers)
        self.assertEqual(created.status_code,201,created.text);ds=created.json()
        self.secrets.append(ds['capability'])
        j=self.client.post('/api/v1/jobs',headers=auth(ds['capability']),json={'dataset_id':ds['dataset_id'],'target':'outcome','positive_label':'event','negative_label':'clear','models':['logistic']})
        self.assertEqual(j.status_code,202,j.text);job=j.json();self.secrets.append(job['capability']);self.jobs.append(job)
        state=self.wait_state(job,('succeeded','partial','failed','timed_out'),timeout=120)
        self.assertEqual(state['state'],'succeeded',state)
        r=self.client.get('/api/v1/jobs/'+job['job_id']+'/result',headers=auth(job['capability'])).json()
        self.assertEqual(r['input_parsing']['details']['import_selection'],{'member':'chosen.csv'})

if __name__=='__main__':unittest.main()
