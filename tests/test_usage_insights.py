"""Analytics attribution, history preservation and honest exclusions. Author: Sherwan Ali."""
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import api_support  # establishes the backend path
import api
import track
import usage
from fastapi.testclient import TestClient


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fyd-usage-')
        self.db = str(Path(self.temp.name) / 'usage.db')
        self.db_patch = patch.object(track, 'DB_PATH', self.db)
        self.key_patch = patch.object(track, 'STATS_KEY', 'fictional-stats-key')
        self.db_patch.start()
        self.key_patch.start()

    def tearDown(self):
        self.key_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def event(self, **context):
        options = usage.request_context({}, '', 'Mozilla/5.0 Chrome/120', '', '/', 200)
        options.update(context)
        with usage.event_context(options):
            track.log_event('visit', ip='192.0.2.1', user_agent='test-browser')

    def test_migration_preserves_six_column_legacy_rows(self):
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute('CREATE TABLE events(ts REAL,day TEXT,kind TEXT,detail TEXT,visitor TEXT,referrer TEXT)')
            c.execute('INSERT INTO events VALUES(?,?,?,?,?,?)', (1,'2020-01-01','visit','legacy','tag','https://www.linkedin.com/feed/'))
        self.event(source='cv')
        data = usage.snapshot(track, 'all')
        self.assertEqual(data['totals'][0], 2)
        self.assertEqual(data['coverage'][0], (1,1))
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('SELECT detail,referrer,metadata_version FROM events ORDER BY ts').fetchone(),
                             ('legacy','https://www.linkedin.com/feed/',0))
        self.assertEqual(usage.snapshot(track, '7')['totals'][0], 1)

    def test_owner_bot_and_signed_tests_are_excluded_but_visible(self):
        self.event()
        self.event(traffic='test')
        self.event(traffic='likely-bot')
        track.log_event('visit', is_owner=1)
        data = usage.snapshot(track, 'all')
        self.assertEqual(data['totals'][0], 1)
        self.assertEqual(sum(row[1] for row in data['traffic']), 4)
        self.assertEqual(usage.request_context({}, '', 'LinkedInBot/1.0', '', '/', 200)['traffic'], 'likely-bot')

    def test_allowlist_rejects_identity_values_and_host_spoofing(self):
        result = usage.attribution({'utm_source':'cv','utm_content':'person@example.com','utm_campaign':'secret-token'})
        self.assertEqual((result['source'],result['content'],result['campaign']), ('cv','',''))
        self.assertEqual(usage.attribution({}, '', 'https://linkedin.com.evil.test/')['source'], 'other-referral')
        self.assertEqual(usage.attribution({}, '', 'https://www.linkedin.com/feed/?secret=x')['source'], 'linkedin')
        self.assertEqual(usage.referrer_host('https://fixyourdataset.fly.dev/stats/private-key'), '')
        self.assertEqual(usage.referrer_host('https://example.test/path?secret=x'), 'example.test')
        self.assertEqual(usage.referrer_host('javascript:alert(1)'), '')
        for header in ('[]', '{"utm_source":[]}', 'broken', '{"utm_source":"cv","utm_content":{}}'):
            self.assertIsInstance(usage.attribution({}, header), dict)

    def test_no_raw_address_agent_or_query_in_database(self):
        with usage.event_context(usage.request_context({'utm_source':'cv'}, '', 'UniqueBrowserString', '', '/', 200)):
            track.log_event('visit', ip='192.0.2.99', user_agent='UniqueBrowserString', referrer='https://example.test/private?token=sensitive')
        with closing(sqlite3.connect(self.db)) as c, c:
            dumped = '\n'.join(c.iterdump())
        for value in ('192.0.2.99','UniqueBrowserString','sensitive','/private'):
            self.assertNotIn(value, dumped)
        track.log_event('analyze', 'private-upload-name.csv folds=secret')
        with closing(sqlite3.connect(self.db)) as c:
            self.assertEqual(c.execute("SELECT detail FROM events WHERE kind='analyze'").fetchone()[0], 'legacy analysis')

    def test_private_dashboard_escapes_history_and_labels_uncertainty(self):
        self.event(source='cv')
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute('UPDATE events SET detail=?', ('<script>alert(1)</script>',))
        page = track.stats_html('all')
        self.assertIn('&lt;script&gt;', page)
        self.assertNotIn('<script>', page)
        for text in ('visitor-days (estimate)', 'not verified people', 'LinkedIn Featured', 'utm_source=cv', 'Direct / unknown'):
            self.assertIn(text,page)
        self.assertEqual(usage.snapshot(track, "30' OR 1=1")['days'], '30')

    def test_new_api_actions_are_counted_without_polling_or_resource_ids(self):
        self.assertEqual(api.usage_action('POST','/api/v1/jobs',202), ('job_submitted',''))
        self.assertEqual(api.usage_action('GET','/api/v1/jobs/private/result',200), ('report_opened',''))
        self.assertIsNone(api.usage_action('GET','/api/v1/jobs/private/result',202))
        self.assertIsNone(api.usage_action('GET','/api/v1/jobs/private',200))
        self.assertIsNone(api.usage_action('HEAD','/',200))
        self.assertEqual(api.usage_action('POST','/api/v1/jobs',422), ('request_refused','job_submitted'))
        self.assertEqual(api.usage_route('/api/v1/jobs/private/result'), '/api/v1/jobs/{id}/result')
        self.assertEqual(api.usage_route('/stats/private-key'), '')

    def test_http_attribution_authentication_and_no_health_noise(self):
        with TestClient(api.create_app()) as client:
            client.get('/?utm_source=cv&utm_medium=resume&utm_campaign=portfolio&utm_content=ml-engineer')
            client.get('/healthz')
            client.head('/')
            client.get('/', headers={'X-FYD-Traffic':'test'})
            client.get('/', headers={'X-FYD-Traffic':'test','X-Stats-Key':'fictional-stats-key'})
            client.get('/', headers={'User-Agent':'LinkedInBot/1.0'})
            # Bad JSON is a refusal, not an accepted job; payload is never retained.
            response = client.post('/api/v1/jobs', content=b'{"private":"not-retained"}', headers={'Content-Type':'application/json',
                'X-FYD-Attribution':json.dumps({'utm_source':'cv','utm_content':'ml-engineer'})})
            self.assertEqual(response.status_code,422)
            self.assertEqual(client.get('/stats').status_code,404)
            page = client.get('/stats',headers={'X-Stats-Key':'fictional-stats-key'},params={'days':'all'})
            self.assertEqual(page.status_code,200)
            self.assertNotIn('fictional-stats-key',page.text)
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('SELECT COUNT(*) FROM events').fetchone()[0],5)
            self.assertEqual(c.execute("SELECT COUNT(*) FROM events WHERE traffic='test'").fetchone()[0],1)
            self.assertEqual(c.execute("SELECT source,content FROM events WHERE kind='request_refused'").fetchone(),('cv','ml-engineer'))
            self.assertNotIn('not-retained','\n'.join(c.iterdump()))

    def test_optional_logging_failure_does_not_escape(self):
        with patch.object(track,'_conn',side_effect=OSError('unavailable')):
            self.assertIsNone(track.log_event('visit'))

    def test_concurrent_first_visits_do_not_lose_events_during_migration(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: track.log_event('visit', user_agent='Mozilla/5.0'), range(18)))
        self.assertEqual(usage.snapshot(track, 'all')['totals'][0],18)


if __name__ == '__main__':
    unittest.main()
