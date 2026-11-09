"""The standalone local server serves every file of the web folder, and only those, by file name."""
import http.client
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from rare_event_eval import server

REAL_WEB = Path(server.__file__).parent / 'web'


class RealWebFolder(unittest.TestCase):
    def test_asset_table_covers_every_web_file_with_a_content_type(self):
        assets = server.web_assets()
        files = {path.name for path in REAL_WEB.iterdir() if path.is_file() and not path.name.startswith('.')}
        self.assertEqual(set(assets), files)
        for name in ('index.html', 'app.js', 'scoring-core.js', 'demo-data.js', 'style.css', 'quality-vs-cost.png',
                     'util.js', 'content.js', 'models.js', 'report-view.js', 'predictions.js', 'api-client.js', 'flow.js', 'views.js'):
            self.assertIn(name, assets)
        self.assertTrue(assets['app.js'].startswith('text/javascript'))
        self.assertTrue(assets['style.css'].startswith('text/css'))
        self.assertEqual(assets['quality-vs-cost.png'], 'image/png')

    def test_export_demo_copies_every_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'demo'
            server.export_demo(target)
            copied = {path.name for path in target.iterdir()}
            self.assertEqual(copied, {path.name for path in REAL_WEB.iterdir()})


class ServedFromATempFolder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        web = Path(cls.tmp.name)
        (web / 'index.html').write_text('<!doctype html><title>x</title>', encoding='utf-8')
        (web / 'app.js').write_text('1', encoding='utf-8')
        (web / 'recorded-example.json').write_text('{"a": 1}', encoding='utf-8')
        (web / 'recorded-example.html').write_text('<!doctype html><style>p{}</style>', encoding='utf-8')
        (web / '.secret').write_text('hidden', encoding='utf-8')
        (web / 'notes.txt').write_text('not served', encoding='utf-8')
        (web / 'sub').mkdir()
        (web / 'sub' / 'inner.js').write_text('2', encoding='utf-8')
        cls.patch = mock.patch.object(server, 'WEB', web)
        cls.patch.start()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.patch.stop()
        cls.tmp.cleanup()

    def get(self, path):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        connection.request('GET', path)
        response = connection.getresponse()
        out = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return out

    def test_served_files_and_content_types(self):
        status, body, headers = self.get('/recorded-example.json')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'a': 1})
        self.assertTrue(headers['Content-Type'].startswith('application/json'))
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(self.get('/')[0], 200)
        self.assertEqual(self.get('/app.js')[0], 200)

    def test_only_listed_regular_files_are_served(self):
        for path in ('/.secret', '/notes.txt', '/sub', '/sub/inner.js', '/..%2fserver.py', '/%2e%2e/server.py', '/static//', '//app.js/', '/app.js/', '/missing.js'):
            self.assertEqual(self.get(path)[0], 404, path)

    def test_the_workbench_keeps_a_strict_policy_and_standalone_pages_get_a_script_free_one(self):
        _, _, page = self.get('/')
        self.assertIn("script-src 'self'", page['Content-Security-Policy'])
        _, _, standalone = self.get('/recorded-example.html')
        self.assertIn("default-src 'none'", standalone['Content-Security-Policy'])
        self.assertIn("style-src 'unsafe-inline'", standalone['Content-Security-Policy'])
        self.assertNotIn('script-src', standalone['Content-Security-Policy'])


if __name__ == '__main__':
    unittest.main()
