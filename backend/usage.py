"""Owner dashboard and bounded request metadata. Author: Sherwan Ali."""
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import json
import re
from urllib.parse import urlsplit

CONTEXT = ContextVar('fyd_usage_context', default=None)
FIELDS = {'metadata_version': 'INTEGER DEFAULT 0', 'source': "TEXT DEFAULT ''", 'medium': "TEXT DEFAULT ''",
          'campaign': "TEXT DEFAULT ''", 'content': "TEXT DEFAULT ''", 'attribution': "TEXT DEFAULT ''",
          'traffic': "TEXT DEFAULT ''", 'device': "TEXT DEFAULT ''", 'browser': "TEXT DEFAULT ''",
          'route': "TEXT DEFAULT ''", 'status': 'INTEGER DEFAULT 0'}
SOURCES = frozenset(('linkedin', 'cv', 'github', 'portfolio', 'google', 'bing', 'duckduckgo'))
MEDIUMS = frozenset(('social', 'resume', 'referral', 'search', 'profile'))
CAMPAIGNS = frozenset(('portfolio', 'applications'))
CONTENTS = frozenset(('featured', 'about', 'project', 'post', 'readme', 'ml-engineer', 'python-backend',
                      'data-science', 'ai-engineer', 'general'))
BOT = re.compile(r'bot\b|crawler|spider|headlesschrome|python-urllib|python-requests|httpx|curl/|wget/', re.I)


def safe_detail(kind, detail):
    if kind != 'analyze':
        return str(detail)[:200]
    # The legacy src/folds fields are caller-controlled: never retain arbitrary
    # names, row values or paths passed through those request fields.
    text = str(detail)
    return text if re.fullmatch(r'(?:demo:(?:liver_disease|credit_default|student_dropout|car_quality)\.csv|upload|test|api|verify|\?) folds=[0-9]{1,3}', text) else 'legacy analysis'


def migrate(c):
    columns = {r[1] for r in c.execute('PRAGMA table_info(events)')}
    for name, definition in FIELDS.items():
        if name not in columns:
            c.execute(f'ALTER TABLE events ADD COLUMN {name} {definition}')
    c.execute('CREATE INDEX IF NOT EXISTS usage_day_kind ON events(day,kind)')
    c.commit()


def referrer_host(value):
    try:
        parts = urlsplit(str(value)[:2048])
        if parts.scheme not in ('http', 'https') or re.search(r'/(?:stats|whoami)(?:/|$)', parts.path):
            return ''
        host = parts.hostname or ''
        return host.lower() if re.fullmatch(r'[a-zA-Z0-9.-]{1,253}', host) else ''
    except ValueError:
        return ''


def attribution(query=None, header='', referrer=''):
    values, evidence = query or {}, 'tagged-link'
    if not values.get('utm_source') and header and len(header) <= 512:
        try:
            parsed = json.loads(header)
            if isinstance(parsed, dict):
                values, evidence = parsed, 'page-context'
        except (ValueError, TypeError):
            pass
    def allowed(key, choices):
        value = values.get(key, '')
        return value if isinstance(value, str) and value in choices else ''
    if allowed('utm_source', SOURCES):
        return {'source': values['utm_source'], 'medium': allowed('utm_medium', MEDIUMS),
                'campaign': allowed('utm_campaign', CAMPAIGNS), 'content': allowed('utm_content', CONTENTS), 'attribution': evidence}
    host = referrer_host(referrer)
    source = 'direct-unknown' if not host or host == 'fixyourdataset.fly.dev' else 'other-referral'
    for domain, name in (('linkedin.com','linkedin'), ('github.com','github'), ('google.com','google'),
                         ('bing.com','bing'), ('duckduckgo.com','duckduckgo')):
        if host == domain or host.endswith('.' + domain):
            source = name
            break
    return {'source': source, 'medium': '', 'campaign': '', 'content': '',
            'attribution': 'unknown' if source == 'direct-unknown' else 'referrer'}


def request_context(query, header, agent, referrer, route, status=200, is_test=False):
    data = attribution(query, header, referrer)
    data.update(metadata_version=1, traffic='test' if is_test else 'likely-bot' if BOT.search(agent) else 'unclassified',
                device='tablet' if re.search('ipad|tablet', agent, re.I) else 'mobile' if re.search('mobile|iphone|android', agent, re.I) else 'desktop-or-other',
                browser=next((name for pattern,name in (('Edg/','Edge'), ('Firefox/','Firefox'), ('Chrome/','Chromium'),
                              ('Safari/','Safari')) if pattern in agent), 'other'), route=route, status=int(status))
    return data


@contextmanager
def event_context(context):
    token = CONTEXT.set(context)
    try:
        yield
    finally:
        CONTEXT.reset(token)


def snapshot(track, days='30'):
    days = str(days) if str(days) in ('7','30','90','all') else '30'
    where, args = '', ()
    if days != 'all':
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=int(days)-1)).isoformat()
        where, args = 'WHERE day >= ?', (cutoff,)
    audience = "is_owner=0 AND COALESCE(traffic,'') NOT IN ('owner','test','likely-bot')"
    condition = (where + ' AND ' if where else 'WHERE ') + audience
    with closing(track._conn()) as c:
        totals = c.execute(f"SELECT SUM(kind='visit'), COUNT(DISTINCT CASE WHEN kind='visit' THEN visitor END), SUM(kind='job_submitted'), SUM(kind='report_opened'), SUM(kind='report_download'), SUM(kind='analyze') FROM events {condition}",args).fetchone()
        queries = {
            'traffic': f"SELECT CASE WHEN is_owner=1 THEN 'owner' WHEN metadata_version=0 THEN 'historical-unclassified' ELSE traffic END,COUNT(*),SUM(kind='visit') FROM events {where} GROUP BY 1 ORDER BY COUNT(*) DESC",
            'sources': f"SELECT CASE WHEN source='' THEN 'historical-unknown' ELSE source END,attribution,medium,campaign,content,SUM(kind='visit'),SUM(kind='job_submitted'),SUM(kind='report_download') FROM events {condition} GROUP BY 1,2,3,4,5 ORDER BY SUM(kind='visit') DESC LIMIT 50",
            'daily': f"SELECT day,SUM(kind='visit'),COUNT(DISTINCT CASE WHEN kind='visit' THEN visitor END),SUM(kind='job_submitted'),SUM(kind='report_download') FROM events {condition} GROUP BY day ORDER BY day DESC LIMIT 90",
            'actions': f'SELECT kind,detail,status,COUNT(*) FROM events {condition} GROUP BY 1,2,3 ORDER BY COUNT(*) DESC LIMIT 40',
            'devices': f"SELECT device,browser,COUNT(*) FROM events {condition} AND kind='visit' AND metadata_version=1 GROUP BY 1,2 ORDER BY COUNT(*) DESC LIMIT 20",
            'referrers': f"SELECT referrer,COUNT(*) FROM events {condition} AND kind='visit' AND referrer!='' GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 20",
            'recent': f"SELECT ts,kind,detail,CASE WHEN is_owner=1 THEN 'owner' WHEN metadata_version=0 THEN 'historical-unclassified' ELSE traffic END,source,attribution,status FROM events {where} ORDER BY ts DESC LIMIT 50",
            'coverage': f'SELECT SUM(metadata_version=1),SUM(metadata_version=0) FROM events {where}'
        }
        result = {key:c.execute(sql,args).fetchall() for key,sql in queries.items()}
    result.update(days=days, totals=[v or 0 for v in totals])
    return result


def render(track, days='all', source='all'):
    from visit_dashboard import render as render_visits
    return render_visits(track, days, source)
