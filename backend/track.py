"""
track.py - minimal, pseudonymous usage logging for Fix Your Dataset.

What is stored per event: timestamp, UTC day, kind (visit / analyze / score), a short detail
string (the demo file name or the word "upload", plus the fold count, for example
"demo:liver_disease.csv folds=1", cut at 200 characters), the HTTP referrer exactly as the
browser sent it (cut at 200 characters, so it can include a query string), an owner flag, and a
pseudonymous visitor tag. Events are written for GET / (kind visit), POST /api/analyze (kind
analyze, written before the analysis runs, so failed runs are logged too) and POST /api/score
(kind score). Nothing is written for sample downloads, /api/profile or /api/convert, and no
route writes a "convert" event.

Not stored: the raw IP address, names, uploaded file contents, file names or column names.
Visitors are not given a cookie (only the optional owner marker below exists).

The visitor tag is the first 12 hex characters of a SHA-256 hash of (ip + user-agent + day +
secret). It changes every day, so it does not link a visitor across days, but it is
pseudonymous, not anonymous. If STATS_KEY is not set the secret is a constant, and in that case
an IP address could be guessed back from a tag if the database ever leaked. Set STATS_KEY on any
real deployment.

Retention: there is no purge job and no deletion route. Rows stay until the database file or the
volume holding it is deleted.

Storage: SQLite. On Fly a persistent volume is mounted at /data so the log survives
deploys and restarts; locally it falls back to ./data_local (gitignored).
"""
import hashlib
import html
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if os.path.isdir("/data"):
    DB_PATH = "/data/usage.db"
else:
    d = _HERE.parent / "data_local"
    d.mkdir(exist_ok=True)
    DB_PATH = str(d / "usage.db")

STATS_KEY = os.environ.get("STATS_KEY", "")   # set as a Fly secret; empty disables /stats


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS events(
        ts REAL, day TEXT, kind TEXT, detail TEXT, visitor TEXT, referrer TEXT)""")
    cols = [r[1] for r in c.execute("PRAGMA table_info(events)").fetchall()]
    if "is_owner" not in cols:                       # tag the site owner's own browser
        c.execute("ALTER TABLE events ADD COLUMN is_owner INTEGER DEFAULT 0")
    return c


def owner_token():
    """Secret value stored in the owner's browser cookie so their own visits can be tagged
    (and kept out of the visitor counts). Derived from STATS_KEY; the raw key is never in the cookie."""
    return hashlib.sha256(f"{STATS_KEY or 'fyd'}|owner-cookie".encode()).hexdigest()[:24]


def visitor_tag(ip, user_agent):
    """Pseudonymous, daily-rotating tag. One-way hash; raw IP is never stored."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = f"{ip}|{user_agent}|{day}|{STATS_KEY or 'fyd'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def log_event(kind, detail="", ip="", user_agent="", referrer="", is_owner=0):
    try:
        now = time.time()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with _conn() as c:
            c.execute("""INSERT INTO events(ts,day,kind,detail,visitor,referrer,is_owner)
                         VALUES (?,?,?,?,?,?,?)""",
                      (now, day, kind, str(detail)[:200],
                       visitor_tag(ip, user_agent), str(referrer)[:200], int(bool(is_owner))))
    except Exception:
        pass   # tracking must never break the app


def _rows(c, q, args=()):
    return c.execute(q, args).fetchall()


def stats_html():
    """The private stats page (HTML)."""
    with _conn() as c:
        # headline numbers EXCLUDE the owner (is_owner=0) so they reflect real visitors
        total_visits = _rows(c, "SELECT COUNT(*) FROM events WHERE kind='visit' AND is_owner=0")[0][0]
        uniq = _rows(c, "SELECT COUNT(DISTINCT visitor) FROM events WHERE kind='visit' AND is_owner=0")[0][0]
        runs = _rows(c, "SELECT COUNT(*) FROM events WHERE kind='analyze' AND is_owner=0")[0][0]
        uploads = _rows(c, """SELECT COUNT(*) FROM events WHERE kind='analyze'
                              AND detail LIKE 'upload%' AND is_owner=0""")[0][0]
        scores = _rows(c, "SELECT COUNT(*) FROM events WHERE kind='score' AND is_owner=0")[0][0]
        mine = _rows(c, "SELECT COUNT(*) FROM events WHERE is_owner=1")[0][0]
        by_day = _rows(c, """SELECT day, SUM(kind='visit'), SUM(kind='analyze')
                             FROM events WHERE is_owner=0 GROUP BY day ORDER BY day DESC LIMIT 30""")
        by_detail = _rows(c, """SELECT detail, COUNT(*) FROM events
                                WHERE kind='analyze' AND is_owner=0
                                GROUP BY detail ORDER BY COUNT(*) DESC LIMIT 15""")
        refs = _rows(c, """SELECT referrer, COUNT(*) FROM events
                           WHERE kind='visit' AND referrer!='' AND is_owner=0 GROUP BY referrer
                           ORDER BY COUNT(*) DESC LIMIT 15""")
        recent = _rows(c, """SELECT ts, kind, detail, visitor, referrer, is_owner FROM events
                             ORDER BY ts DESC LIMIT 40""")

    def tbl(headers, rows):
        h = "<tr>" + "".join(f"<th>{x}</th>" for x in headers) + "</tr>"
        b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>"
                    for r in rows)
        return f"<table>{h}{b}</table>"

    recent_fmt = [(datetime.fromtimestamp(ts, timezone.utc).strftime("%m-%d %H:%M"),
                   k + (" (you)" if o else ""), d, v, r) for ts, k, d, v, r, o in recent]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Fix Your Dataset - stats</title>
<style>body{{font:14px/1.5 system-ui;margin:40px auto;max-width:880px;color:#1a1a1a;padding:0 16px}}
h1{{font-size:22px}} h2{{font-size:15px;margin-top:28px;color:#0E7A5F}}
.big{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
.card{{border:1px solid #E4E2DD;border-radius:10px;padding:12px 18px}}
.card b{{font-size:26px;display:block;font-family:ui-monospace,monospace}}
.card span{{color:#6B7280;font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
th,td{{border-bottom:1px solid #E4E2DD;padding:6px 9px;text-align:left}}
th{{color:#6B7280;font-size:11px;text-transform:uppercase}}</style></head><body>
<h1>Fix Your Dataset - private stats</h1>
<div class="big">
 <div class="card"><b>{total_visits}</b><span>page visits</span></div>
 <div class="card"><b>{uniq}</b><span>unique visitors (daily-anon)</span></div>
 <div class="card"><b>{runs}</b><span>analyses run</span></div>
 <div class="card"><b>{uploads}</b><span>own files uploaded</span></div>
 <div class="card"><b>{scores}</b><span>files scored</span></div>
</div>
<p style="color:#6B7280;font-size:12px">The counts above exclude your own browser
({mine} event(s) tagged as you). Mark a browser as yours by opening
<code>/whoami/&lt;key&gt;</code> on the site. Your rows show "(you)" in the log below.</p>
<h2>Last 30 days (visits / analyses)</h2>{tbl(['day','visits','analyses'], by_day)}
<h2>What people run</h2>{tbl(['what','times'], by_detail)}
<h2>Where visitors come from (referrers)</h2>{tbl(['referrer','visits'], refs)}
<h2>Recent activity (UTC)</h2>{tbl(['time','kind','detail','visitor','referrer'], recent_fmt)}
<p style="color:#6B7280;font-size:12px">Usage log: each event stores a time, the action, the demo
name (or the word upload), the referrer as sent, and a pseudonymous visitor tag that changes daily.
No raw IPs, names or file contents are stored and visitors get no cookie. The tag is pseudonymous,
not anonymous (and weaker if STATS_KEY is not set). Events are never deleted automatically. The only
cookie is an optional owner marker you set yourself, used just to keep your own visits out of these
numbers.</p>
</body></html>"""
