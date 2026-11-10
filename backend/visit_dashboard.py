"""Visits and entry sources for the private owner page. Author: Sherwan Ali."""
from contextlib import closing
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import html
import re
from urllib.parse import urlencode
import usage

BERLIN = ZoneInfo('Europe/Berlin')
CHANNELS = {'cv':'CV', 'linkedin':'LinkedIn', 'github':'GitHub', 'search':'Search',
            'other':'Other websites', 'unknown':'Direct / unknown'}
CV_LABELS = {'python-backend':'Software CV', 'ml-engineer':'ML Research CV',
             'ai-engineer':'AI / GenAI CV', 'general':'German CV', 'data-science':'Data science CV'}
PLACEMENTS = {'featured':'Featured card', 'about':'About section', 'project':'Project link',
              'post':'Existing post', 'readme':'README link'}
AUDIENCE = "is_owner=0 AND COALESCE(traffic,'') NOT IN ('owner','test','likely-bot')"


def host_of(value):
    value = str(value or '')
    if re.fullmatch(r'[A-Za-z0-9.-]{1,253}', value):
        return value.lower()
    return usage.referrer_host(value)


def channel_of(source, referrer):
    """Read historical referrers without changing or guessing missing records."""
    if source not in usage.SOURCES:
        host = host_of(referrer)
        source = usage.attribution({}, '', 'https://' + host if host else '')['source']
    if source in ('cv','linkedin','github'):
        return source
    if source in ('google','bing','duckduckgo'):
        return 'search'
    return 'unknown' if source == 'direct-unknown' else 'other'


def origin_detail(channel, content, referrer):
    if channel == 'cv':
        return CV_LABELS.get(content, 'CV link')
    if channel in ('linkedin','github'):
        return PLACEMENTS.get(content, host_of(referrer) or CHANNELS[channel] + ' link')
    if channel in ('other','search'):
        return host_of(referrer) or ('Search link' if channel=='search' else 'Other website link')
    return 'No source was provided'


def snapshot(track, days='all', source='all'):
    days = str(days) if str(days) in ('7','30','90','all') else 'all'
    source = source if source in CHANNELS else 'all'
    date_sql, args = '', ()
    if days != 'all':
        day = datetime.now(BERLIN).date() - timedelta(days=int(days)-1)
        cutoff = datetime.combine(day, datetime.min.time(), tzinfo=BERLIN).timestamp()
        date_sql, args = ' AND ts >= ?', (cutoff,)
    visits = "kind='visit'" + date_sql
    with closing(track._conn()) as c:
        c.create_function('entry_channel', 2, channel_of)
        counts = dict(c.execute(f'SELECT entry_channel(source,referrer),COUNT(*) FROM events WHERE {visits} AND {AUDIENCE} GROUP BY 1',args))
        excluded = dict(c.execute(f"SELECT CASE WHEN is_owner=1 THEN 'owner' ELSE traffic END,COUNT(*) FROM events WHERE {visits} AND NOT ({AUDIENCE}) GROUP BY 1",args))
        selection = '' if source == 'all' else ' AND entry_channel(source,referrer)=?'
        recent = c.execute(f'SELECT ts,entry_channel(source,referrer),content,referrer,attribution FROM events WHERE {visits} AND {AUDIENCE}{selection} ORDER BY ts DESC,rowid DESC LIMIT 100', args + (() if source=='all' else (source,))).fetchall()
        excluded_recent = c.execute(f"SELECT ts,CASE WHEN is_owner=1 THEN 'owner' ELSE traffic END,entry_channel(source,referrer),content,referrer FROM events WHERE {visits} AND NOT ({AUDIENCE}) ORDER BY ts DESC,rowid DESC LIMIT 20",args).fetchall()
    return {'days':days,'source':source,'counts':{key:counts.get(key,0) for key in CHANNELS},
            'total':sum(counts.values()),'excluded':excluded,'recent':recent,'excluded_recent':excluded_recent}


def render(track, days='all', source='all'):
    data=snapshot(track, days, source)
    esc=lambda value:html.escape(str(value),quote=True)
    def link(period=None, channel=None):
        return '?' + urlencode({'days':period or data['days'],'source':channel or data['source']})
    def stamp(ts):
        return datetime.fromtimestamp(ts,BERLIN).strftime('%d %b %Y, %H:%M %Z')
    def table(headers, rows, empty):
        head=''.join('<th scope="col">'+esc(h)+'</th>' for h in headers)
        body=''.join('<tr>'+''.join('<td>'+esc(cell)+'</td>' for cell in row)+'</tr>' for row in rows)
        if not body:body=f'<tr><td colspan="{len(headers)}" class="empty">{esc(empty)}</td></tr>'
        return '<div class="table-wrap"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'
    nav=''.join(f'<a href="{esc(link(period=period))}" aria-current="{"page" if period==data["days"] else "false"}">{label}</a>' for period,label in (('7','7 days'),('30','30 days'),('90','90 days'),('all','All time')))
    cards=''
    for key,label in CHANNELS.items():
        count=data['counts'][key];share=100*count/data['total'] if data['total'] else 0
        cards+=f'<a class="source-card" href="{esc(link(channel=key))}" aria-current="{"page" if key==data["source"] else "false"}"><span>{esc(label)}</span><strong>{count}</strong><small>{share:.1f}% of visits</small></a>'
    rows=[]
    for ts,channel,content,referrer,evidence in data['recent']:
        how='Tagged link' if evidence in ('tagged-link','page-context') else 'Referring website' if channel!='unknown' and host_of(referrer) else 'Not available'
        rows.append((stamp(ts),CHANNELS[channel],origin_detail(channel,content,referrer),how))
    traffic_names={'owner':'You','test':'Marked testing','likely-bot':'Likely automated'}
    excluded_rows=[(stamp(ts),traffic_names.get(traffic,'Excluded'),CHANNELS[channel],origin_detail(channel,content,referrer)) for ts,traffic,channel,content,referrer in data['excluded_recent']]
    exclusions=' · '.join(f'{traffic_names[key]}: {data["excluded"].get(key,0)}' for key in traffic_names)
    title='Recent visits' if data['source']=='all' else 'Recent visits from '+CHANNELS[data['source']]
    filtered='' if data['source']=='all' else f'<a class="clear" href="{esc(link(channel="all"))}">Show all sources</a>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visit sources | FixYourDataset</title>
<style>body{{font:15px/1.55 system-ui;margin:0;background:#f5f7f9;color:#172a34}}main{{max-width:1120px;margin:40px auto;padding:0 22px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:start}}.eyebrow{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#566a75;margin:0}}h1{{font-size:32px;margin:7px 0}}.intro{{color:#526773;max-width:700px;margin:0 0 24px}}.total{{text-align:right;white-space:nowrap}}.total strong{{display:block;font-size:42px;line-height:1.2}}.total span{{color:#526773;font-size:13px}}nav{{display:flex;gap:8px;margin:16px 0 22px;flex-wrap:wrap}}nav a,.clear{{color:#175a75;text-decoration:none;border:1px solid #c6d5df;padding:7px 14px;border-radius:7px;background:white}}nav a[aria-current=page]{{background:#163f55;color:white;border-color:#163f55}}.sources{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.source-card{{text-decoration:none;color:inherit;background:white;border:1px solid #d8e1e8;border-radius:12px;padding:17px 20px;display:flex;flex-direction:column}}.source-card:hover,.source-card[aria-current=page]{{border-color:#237396;box-shadow:0 0 0 1px #237396}}.source-card span{{font-weight:600}}.source-card strong{{font-size:32px;margin:3px 0}}.source-card small{{color:#647785}}.hint{{font-size:13px;color:#526773;margin:13px 0 28px}}.section-head{{display:flex;justify-content:space-between;gap:12px;align-items:center}}h2{{font-size:20px;margin:0}}.section-note{{font-size:13px;color:#647785;margin:5px 0 14px}}.table-wrap{{overflow:auto;border:1px solid #d8e1e8;border-radius:10px;background:white}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:12px 15px;border-bottom:1px solid #e6ecf0;overflow-wrap:anywhere}}th{{font-size:12px;color:#617785;background:#edf2f6;white-space:nowrap}}tr:last-child td{{border-bottom:0}}td:first-child{{white-space:nowrap}}.empty{{color:#647785;padding:28px}}details{{margin:24px 0;padding:15px 18px;background:#eaf0f4;border-radius:10px}}summary{{font-weight:600;cursor:pointer}}details p,footer{{font-size:13px;color:#526773}}details .table-wrap{{margin-top:15px}}footer{{margin:25px 0 40px;max-width:880px}}@media(max-width:640px){{main{{margin-top:24px;padding:0 14px}}h1{{font-size:25px}}header{{gap:12px}}.total strong{{font-size:32px}}.sources{{grid-template-columns:repeat(2,minmax(0,1fr))}}.source-card{{padding:14px}}.section-head{{align-items:start;flex-direction:column}}th,td{{padding:10px}}}}</style></head><body><main>
<header><div><p class="eyebrow">FixYourDataset · private stats</p><h1>Where visits come from</h1><p class="intro">See whether someone opened your CV link, LinkedIn link, GitHub link or another website link.</p></div><div class="total"><strong>{data['total']}</strong><span>visits in this period</span></div></header>
<nav aria-label="Visit date range">{nav}</nav><section class="sources" aria-label="Visits by source">{cards}</section>
<p class="hint">Select a source to see its visits below. Your marked visits, testing and likely automated traffic are excluded.</p>
<section aria-labelledby="recent-heading"><div class="section-head"><h2 id="recent-heading">{esc(title)}</h2>{filtered}</div><p class="section-note">Latest 100 visits. Times shown in Berlin local time.</p>{table(['When','Came from','Which link or website','How we know'],rows,'No visits recorded for this source in this period.')}</section>
<details><summary>Your visits and testing</summary><p>{esc(exclusions)}. These visits are not included above.</p>{table(['When','Type','Came from','Link or website'],excluded_rows,'No excluded visits in this period.')}<p>Latest 20 excluded visits. Likely automated traffic is detected by a heuristic; it does not identify every bot.</p></details>
<footer>CV and LinkedIn tags show which link was used, not who clicked it. A forwarded CV link still counts as CV. A missing source stays <strong>Direct / unknown</strong>; it can be an untagged CV, bookmark, copied link or a browser that omitted its referrer. Older visits use their saved referrer where available. Reloads count as another visit. No names, raw IP addresses or uploaded data are shown. Date ranges use Berlin calendar days.</footer>
</main></body></html>'''
