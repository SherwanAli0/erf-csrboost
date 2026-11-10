"""Owner dashboard and bounded request metadata. Author: Sherwan Ali."""
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import html
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


def render(track, days='30'):
    data = snapshot(track, days)
    def table(headers, rows):
        head = ''.join('<th scope="col">'+html.escape(x)+'</th>' for x in headers)
        body = ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)
        if not body:
            body = f'<tr><td colspan="{len(headers)}">No recorded events in this period.</td></tr>'
        return '<div class="table-wrap"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'
    cards = ''.join(f'<div class="card"><b>{value}</b><span>{label}</span></div>' for value,label in zip(data['totals'],
        ('page requests','visitor-days (estimate)','jobs accepted','report retrievals','report downloads','legacy analysis attempts')))
    nav = ''.join(f'<a href="?days={period}" aria-current="{"page" if period==data["days"] else "false"}">{label}</a> ' for period,label in (('7','7 days'),('30','30 days'),('90','90 days'),('all','All time')))
    recent = [(datetime.fromtimestamp(r[0],timezone.utc).strftime('%Y-%m-%d %H:%M'),*r[1:]) for r in data['recent']]
    # Historical URLs stay intact on disk, but private paths/queries are never echoed.
    refs = [(referrer_host(r) or (r if re.fullmatch(r'[A-Za-z0-9.-]{1,253}', r) else 'omitted'), n) for r,n in data['referrers']]
    presets = [('LinkedIn Featured','linkedin','social','featured'),('CV: ML engineer','cv','resume','ml-engineer'),
               ('CV: Python backend','cv','resume','python-backend'),('CV: data science','cv','resume','data-science'),
               ('CV: AI engineer','cv','resume','ai-engineer'),('GitHub README','github','referral','readme')]
    links = [(label,f'https://fixyourdataset.fly.dev/?utm_source={source}&utm_medium={medium}&utm_campaign=portfolio&utm_content={content}') for label,source,medium,content in presets]
    coverage = [n or 0 for n in data['coverage'][0]]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FixYourDataset private stats</title>
<style>body{{font:15px/1.55 system-ui;margin:36px auto;max-width:1150px;padding:0 18px;color:#182b30;background:#f7f9fa}}h1{{font-size:28px}}h2{{font-size:19px;margin-top:32px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{padding:16px;border:1px solid #cddbdc;background:white;border-radius:10px;min-width:130px}}.card b{{display:block;font-size:29px}}.card span{{font-size:13px}}.note{{padding:16px;background:#e6f1f0;border-radius:8px}}nav a{{display:inline-block;margin:5px;padding:7px 12px;border:1px solid #799c9b;border-radius:6px;color:#124e4a}}[aria-current=page]{{background:#124e4a;color:white}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #d8e2e3;max-width:560px;overflow-wrap:anywhere}}.table-wrap{{overflow:auto}}footer{{margin:30px 0;color:#435b62}}</style></head><body>
<h1>FixYourDataset private stats</h1><p>UTC dates. Period: {html.escape(data['days'])}. Refresh to update counts.</p><nav aria-label="Date range">{nav}</nav><div class="cards">{cards}</div>
<p class="note">Main counts exclude marked owner, authenticated test and likely-bot traffic. The remainder is unclassified traffic, not verified people. One person can generate multiple requests and one visitor tag per day or device. Accepted jobs do not prove completed analyses; report retrievals and downloads can repeat.</p>
<h2>Sources and tagged links</h2><p>Source is available when a recognised campaign tag or referrer is present. Direct / unknown includes missing referrers, untagged CVs, privacy tools and copied links. Tags describe the link used, not a person's identity or occupation. Forwarded links retain their tag.</p>
{table(['Source','Evidence','Medium','Campaign','Placement / CV','Page requests','Jobs accepted','Report downloads'],data['sources'])}
<h2>Owner, tests and likely bots</h2>{table(['Traffic class','All events','Page requests'],data['traffic'])}<p>Owner exclusion requires the existing owner marker on each browser. Automated checks use an authenticated test header. Bot detection is a heuristic. Historical tests cannot be reliably recovered.</p>
<h2>Daily usage</h2>{table(['UTC day','Page requests','Visitor-days','Jobs accepted','Report downloads'],data['daily'])}
<h2>Actions and HTTP outcomes</h2>{table(['Action','Fixed detail','HTTP status (0 = legacy)','Requests'],data['actions'])}<p>New workbench uploads, jobs, report retrievals, aggregate exports and data-tool responses are counted server-side. Reading tabs, opening an example and browser-only prediction evaluation are not tracked. No session duration or identity is inferred.</p>
<h2>Device and browser families</h2>{table(['Device family','Browser family','Page requests'],data['devices'])}<p>Coarse user-agent inference only. Chromium includes Brave; raw user-agent strings are not retained.</p>
<h2>Referrers</h2>{table(['Referrer host','Page requests'],refs)}
<h2>Recent activity, including excluded traffic</h2>{table(['UTC time','Action','Detail','Traffic','Source','Evidence','HTTP status'],recent)}
<h2>Links to use from now on</h2>{table(['Placement','Public destination'],links)}<p>CV and LinkedIn link edits are separate publication steps; these examples are not applied automatically.</p>
<footer>Coverage: {coverage[0]} events with new metadata, {coverage[1]} historical events. Historical rows are preserved, not retroactively classified. New records contain time, fixed action/route/status, recognised campaign labels, referrer host, coarse device/browser class, owner/test class and a daily pseudonymous tag. No raw IP, uploaded content or visitor cookie is stored. The optional owner cookie remains. Retention is unchanged; records are not automatically deleted.</footer></body></html>'''
