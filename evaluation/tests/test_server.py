import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from rare_event_eval.server import Handler, export_demo


class LocalAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def test_health_head_and_asset_allowlist(self):
        status, body, _ = self.request('GET', '/api/health')
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)['uploads_stored'])
        self.assertEqual(self.request('HEAD', '/')[1], b'')
        self.assertEqual(self.request('GET', '/../server.py')[0], 404)
        self.assertEqual(self.request('GET', '/quality-vs-cost.png')[0], 200)

    def test_foreign_hosts_and_origins_rejected(self):
        self.assertEqual(self.request('GET', '/', headers={'Host': 'example.org'})[0], 403)
        self.assertEqual(self.request('GET', '/', headers={'Origin': 'https://example.org'})[0], 403)

    def test_body_validation(self):
        self.assertEqual(self.request('POST', '/api/evaluate', '{}')[0], 415)
        headers = {'Content-Type': 'application/json'}
        self.assertEqual(self.request('POST', '/api/evaluate', '{', headers)[0], 400)
        self.assertEqual(self.request('POST', '/api/evaluate', '{}', headers)[0], 400)
        self.assertEqual(self.request('POST', '/api/evaluate', '', {**headers, 'Content-Length': '2000001'})[0], 413)

    def test_prediction_api_returns_aggregates(self):
        payload = dict(csv='outcome,score\nclear,0.1\nevent,0.9\n', label_column='outcome',
                       score_columns=['score'], positive_label='event', negative_label='clear',
                       score_kind='probability', review_budgets=[1])
        status, body, _ = self.request('POST', '/api/evaluate', json.dumps(payload), {'Content-Type': 'application/json'})
        self.assertEqual(status, 200)
        report = json.loads(body)
        self.assertEqual(report['models']['score']['metrics']['average_precision'], 1)
        self.assertIn('input_sha256', report['data'])
        self.assertNotIn('test_truth', report)
        self.assertNotIn('test_scores', report)

    def test_export_includes_assets_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'demo'
            export_demo(target)
            for name in ('index.html', 'app.js', 'scoring-core.js', 'demo-data.js', 'style.css', 'quality-vs-cost.png'):
                self.assertTrue((target/name).is_file())
            with self.assertRaises(ValueError):
                export_demo(target)
