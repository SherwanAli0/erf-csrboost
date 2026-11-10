"""First-party pseudonymous usage logging. Author: Sherwan Ali.

New records keep daily salted visitor estimates, fixed actions, recognised
campaign labels, referrer hosts, coarse device/browser families and traffic
classification. Raw IPs, user agents, uploaded contents and arbitrary query
values are not stored. Historical rows and retention are unchanged. No
visitor cookies; the optional owner cookie remains. See docs/USAGE-STATS.md.
"""
import hashlib
import os
import sqlite3
import time
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import usage

_HERE = Path(__file__).resolve().parent
if os.path.isdir("/data"):
    DB_PATH = "/data/usage.db"
else:
    d = _HERE.parent / "data_local"
    d.mkdir(exist_ok=True)
    DB_PATH = str(d / "usage.db")

STATS_KEY = os.environ.get("STATS_KEY", "")   # set as a Fly secret; empty disables /stats
_SCHEMA_LOCK = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=10)
    try:
        with _SCHEMA_LOCK:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS events(
                ts REAL, day TEXT, kind TEXT, detail TEXT, visitor TEXT, referrer TEXT)""")
            cols = [r[1] for r in c.execute("PRAGMA table_info(events)").fetchall()]
            if "is_owner" not in cols:
                c.execute("ALTER TABLE events ADD COLUMN is_owner INTEGER DEFAULT 0")
            usage.migrate(c)
        return c
    except Exception:
        c.close()
        raise


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
        context = usage.CONTEXT.get() or usage.request_context({}, '', user_agent, referrer, '', 0)
        values = {name: context.get(name, 0 if name in ('metadata_version', 'status') else '') for name in usage.FIELDS}
        if is_owner:
            values['traffic'] = 'owner'
        with closing(_conn()) as c, c:
            c.execute(f"INSERT INTO events(ts,day,kind,detail,visitor,referrer,is_owner,{','.join(values)}) "
                      f"VALUES ({','.join('?' for _ in range(7+len(values)))})",
                      (now, day, kind, usage.safe_detail(kind, detail), visitor_tag(ip, user_agent),
                       usage.referrer_host(referrer), int(bool(is_owner)), *values.values()))
    except Exception:
        pass   # tracking must never break the app


def stats_html(days='all', source='all'):
    import sys
    return usage.render(sys.modules[__name__], days, source)
