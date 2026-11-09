"""Behaviour checks for explicit-label tools and opt-in row exports. Author: Sherwan Ali."""
import csv
import io
import unittest

import api_support
import numpy as np
from services.toolbox import execute


def fixture():
    out=io.StringIO(); writer=csv.writer(out,lineterminator='\n')
    writer.writerow(['customer_id','value','region','outcome'])
    for i in range(160): writer.writerow([f'person-{i}', i/7, 'north' if i%2 else 'south', 'event' if i%8==0 else 'clear'])
    return out.getvalue()


def payload(**overrides):
    result=dict(csv=fixture(), target='outcome', positive_label='event', negative_label='clear',
                exclude_columns=['customer_id'],seed=42,confirm_labels=True,operation='resample',method='smote')
    result.update(overrides)
    return result


class ToolContract(unittest.TestCase):
    def test_no_row_data_without_explicit_opt_in(self):
        result=execute(payload())
        self.assertEqual(result['downloads'],{})
        self.assertNotIn('person-0',str(result))
        self.assertFalse(result['row_data_included'])

    def test_exports_have_disjoint_partitions_and_truthful_mapping(self):
        result=execute(payload(include_row_data=True))
        files=result['downloads']
        indices=list(csv.DictReader(io.StringIO(files['split_indices.csv'])))
        self.assertEqual(len(indices),160)
        self.assertEqual(len({x['source_row'] for x in indices}),160)
        source=list(csv.DictReader(io.StringIO(fixture())))
        for key,file in [('validation','validation.csv'),('test','heldout_test.csv')]:
            expected=[source[int(x['source_row'])] for x in indices if x['partition']==key]
            self.assertEqual(list(csv.DictReader(io.StringIO(files[file]))),expected)
        train=list(csv.DictReader(io.StringIO(files['resampled_train.csv'])))
        self.assertNotIn('customer_id',train[0])
        self.assertEqual({x['region'] for x in train},{'north','south'})
        mapping=list(csv.DictReader(io.StringIO(files['row_mapping.csv'])))
        self.assertEqual(len(mapping),len(train))
        synthetic=[x for x in mapping if x['origin']=='synthetic']
        self.assertTrue(synthetic)
        self.assertTrue(all(not x['source_row_candidates'] for x in synthetic))

    def test_confirmed_majority_positive_is_not_silently_flipped(self):
        body=payload();body.update(positive_label='clear',negative_label='event')
        result=execute(body)
        self.assertEqual(result['positive_label'],'clear')
        self.assertGreater(result['train_counts_before'][1],result['train_counts_before'][0])
        body['confirm_labels']=False
        with self.assertRaisesRegex(ValueError,'Confirm'):execute(body)

    def test_score_refit_returns_no_evaluation_claim_and_does_not_overwrite_columns(self):
        result=execute(payload(operation='score',score_csv='value,region,fyd_positive_score\n1,west,original\n2,north,original2\n',include_row_data=True,model='logistic'))
        rows=list(csv.DictReader(io.StringIO(result['downloads']['scored_rows.csv'])))
        self.assertEqual(rows[0]['fyd_positive_score'],'original')
        self.assertIn('_fyd_positive_score',rows[0])
        self.assertEqual(result['rows'],2)
        self.assertIn('no evaluation metrics',result['fit_scope'])
        self.assertNotIn('report',result)

    def test_leakage_arms_share_exact_original_test_rows(self):
        result=execute(payload(operation='leakage',include_row_data=True))
        rows=list(csv.DictReader(io.StringIO(result['downloads']['heldout_scores.csv'])))
        original=list(csv.DictReader(io.StringIO(fixture())))
        self.assertTrue(result['same_original_test_population'])
        self.assertEqual(len(rows),result['evaluation_rows'])
        self.assertTrue(all(r['label']==original[int(r['source_row'])]['outcome'] for r in rows))
        self.assertEqual(set(result['report']['models']),{'train_only_sampler','contaminated_sampler'})
        self.assertTrue(all(np.isfinite(float(r['contaminated_sampler'])) for r in rows))

    def test_invalid_request_never_falls_back_or_drops_rows(self):
        for change in ({'method':'unknown'},{'negative_label':'missing'},{'seed':True},{'exclude_columns':['absent']}):
            body=payload();body.update(change)
            with self.assertRaises(ValueError): execute(body)

    def test_formula_strings_are_safe_but_negative_numbers_are_preserved(self):
        from services.toolbox import _csv
        import pandas as pd
        text=_csv(pd.DataFrame({'=header':['=cmd','-1',-2,'+SUM(1,2)','@x']}))
        rows=list(csv.reader(io.StringIO(text)))
        self.assertEqual(rows[0],["'=header"])
        self.assertEqual([r[0] for r in rows[1:]],["'=cmd",'-1','-2',"'+SUM(1,2)","'@x"])


class ToolHTTP(api_support.ApiCase):
    def test_real_bounded_child_and_no_default_row_download(self):
        response=self.client.post('/api/v1/tools/resample',json=payload())
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['downloads'],{})

    def test_opt_in_file_returns_only_to_the_calling_response(self):
        response=self.client.post('/api/v1/tools/resample',json=payload(include_row_data=True))
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn('heldout_test.csv',response.json()['downloads'])
        self.assertEqual(self.client.get('/api/v1/tools/resample/result').status_code,404)

    def test_error_contract_media_type_and_validation(self):
        unknown=self.client.post('/api/v1/tools/missing',json={})
        self.assertEqual(unknown.status_code,404)
        self.assertEqual(unknown.json()['error']['code'],'TOOL_UNKNOWN')
        self.assertEqual(self.client.post('/api/v1/tools/resample',content='{}',headers={'Content-Type':'text/plain'}).status_code,415)
        self.assertEqual(self.client.post('/api/v1/tools/resample',json=[]).status_code,422)
        body=payload();body['confirm_labels']=False
        self.assertEqual(self.client.post('/api/v1/tools/resample',json=body).status_code,400)


if __name__=='__main__':unittest.main()
