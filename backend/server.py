"""
server.py - standard library HTTP server for FixYourDataset (no web framework needed).

Routes (HEAD is answered like GET, with the same headers and no body):
  GET  /                 -> frontend/index.html
  GET  /healthz          -> "ok" (also /health); used as the health check
  GET  /api/samples      -> list of bundled sample datasets
  GET  /api/sample?name= -> raw CSV text of a sample
  GET  /stats            -> owner-only usage page (key as ?key= or /stats/<key>); 404 without the key
  GET  /whoami/<key>     -> marks this browser as the owner; 404 without the key
  POST /api/profile      -> {csv} -> column profile + suggested target
  POST /api/analyze      -> {csv,target,method,do_classify,classifier,folds,src} -> full leakage-safe result
  POST /api/score        -> {train_csv,target,method,classifier,score_csv} -> risk score for each new row
  POST /api/convert      -> {b64,name} -> CSV text and notes (decodes the text encoding, unpacks
                            ZIP/GZIP, converts Excel, JSON and Stata files; 30 MB limit)

Run from the repository root:  python backend/server.py  (py -3.11 backend/server.py on Windows),
then open http://localhost:8600. The PORT environment variable sets another port.
The server listens on every network interface (0.0.0.0), not only on localhost.
Deploy: the Dockerfile runs the same code and binds to $PORT; fly.toml and DEPLOY_FLY.md cover Fly.io.
"""
import json, os, sys, warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine  # noqa: E402
import track   # noqa: E402

ROOT = HERE.parent
INDEX = ROOT / "frontend" / "index.html"
SAMPLES = ROOT / "samples"
PORT = int(os.environ.get("PORT", 8600))

SAMPLE_META = {
    "liver_disease.csv":   {"category": "Medical",        "emoji": "\U0001FA7A",
                            "label": "Liver disease",     "desc": "find the less common group: records without liver disease (583 rows, 2.5:1)"},
    "credit_default.csv":  {"category": "Finance",        "emoji": "\U0001F4B3",
                            "label": "Credit-card default", "desc": "predict who defaults next month (6,000 rows, 3.5:1)"},
    "student_dropout.csv": {"category": "Education",      "emoji": "\U0001F393",
                            "label": "Student dropout",   "desc": "find the less common group: currently enrolled students (4,424 rows, 4.6:1)"},
    "car_quality.csv":     {"category": "Manufacturing",  "emoji": "\U0001F697",
                            "label": "Car quality",       "desc": "catch the rare 'good' cars (1,728 rows, 24:1)"},
}
SAMPLE_ORDER = ["Medical", "Finance", "Education", "Manufacturing"]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, extra_headers=None):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":   # HEAD: same status and headers as GET, no body
            self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    MAX_BODY = 45 * 1024 * 1024   # a 30MB upload is ~40MB base64 + JSON; anything past 45MB is abuse

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n < 0:
            raise ValueError("That request has an invalid size.")
        if n > self.MAX_BODY:
            raise ValueError("That request is too large. Please upload a file under 30 MB.")
        return json.loads(self.rfile.read(n) or b"{}")

    def _who(self):
        """(ip, user_agent, referrer) - ip is used only for the daily pseudonymous hash and is not stored."""
        ip = (self.headers.get("Fly-Client-IP")
              or (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
              or self.client_address[0])
        return ip, self.headers.get("User-Agent", ""), self.headers.get("Referer", "")

    def _is_owner(self):
        """True if this browser carries the owner cookie (set via /whoami/<key>)."""
        from http.cookies import SimpleCookie
        raw = self.headers.get("Cookie", "")
        if not raw:
            return False
        try:
            got = SimpleCookie(raw).get("fyd_owner")
            return bool(got and got.value == track.owner_token())
        except Exception:
            return False

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/healthz", "/health"):
            self._send(200, "ok", "text/plain")
        elif path == "/stats" or path.startswith("/stats/"):
            from urllib.parse import parse_qs, urlparse
            key = parse_qs(urlparse(self.path).query).get("key", [""])[0]
            if not key and path.startswith("/stats/"):
                key = path.split("/stats/", 1)[1].strip("/")   # also accept /stats/<key>
            if track.STATS_KEY and key == track.STATS_KEY:
                self._send(200, track.stats_html(), "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")
        elif path.startswith("/whoami/"):
            key = path.split("/whoami/", 1)[1].strip("/")
            if track.STATS_KEY and key == track.STATS_KEY:
                cookie = (f"fyd_owner={track.owner_token()}; Max-Age=31536000; "
                          f"Path=/; SameSite=Lax")
                self._send(200, "<!doctype html><meta charset=utf-8><body style='font:16px system-ui;"
                           "max-width:520px;margin:60px auto;padding:0 16px'><h2>This browser is now "
                           "marked as you.</h2><p>Your visits and runs will be tagged \"(you)\" and "
                           "kept out of the visitor counts. Repeat this on any other device you use. "
                           "To undo, clear this site's cookies.</p><p><a href='/'>Go to the app</a></p>",
                           "text/html; charset=utf-8", {"Set-Cookie": cookie})
            else:
                self._send(404, "not found", "text/plain")
        elif path in ("/", "/index.html"):
            ip, ua, ref = self._who()
            if self.command != "HEAD":   # uptime probes must not count as visits
                track.log_event("visit", "", ip, ua, ref, is_owner=self._is_owner())
            self._send(200, INDEX.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        elif path == "/api/samples":
            items = [{"name": n, **SAMPLE_META[n]} for n in SAMPLE_META if (SAMPLES / n).exists()]
            self._json(200, {"samples": items, "order": SAMPLE_ORDER})
        elif path == "/api/sample":
            from urllib.parse import parse_qs, urlparse
            name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
            f = SAMPLES / name
            if name in SAMPLE_META and f.exists():
                self._send(200, f.read_text(encoding="utf-8"), "text/csv")
            else:
                self._send(404, "not found", "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            b = self._body()
            if path == "/api/profile":
                df = engine.read_csv(b["csv"])
                self._json(200, engine.profile(df))
            elif path == "/api/analyze":
                target = b.get("target")
                if not target:
                    raise ValueError("Could not read that file well enough to find a target column. "
                                     "Check the file's format.")
                ip, ua, ref = self._who()
                track.log_event("analyze",
                                f"{b.get('src', '?')} folds={b.get('folds', 1)}", ip, ua, ref,
                                is_owner=self._is_owner())
                folds = max(1, min(100, int(b.get("folds", 1))))   # clamp: one POST can't pin the CPU
                r = engine.analyze(b["csv"], target, b.get("method", "smote"),
                                   do_classify=bool(b.get("do_classify")),
                                   classifier=b.get("classifier", "logreg"),
                                   folds=folds)
                self._json(200, r)
            elif path == "/api/score":
                ip, ua, ref = self._who()
                track.log_event("score", "", ip, ua, ref, is_owner=self._is_owner())
                r = engine.score_new(b["train_csv"], b["target"], b.get("method", "smote"),
                                     b.get("classifier", "random_forest"), b["score_csv"])
                self._json(200, r)
            elif path == "/api/convert":
                import base64
                raw = base64.b64decode(b["b64"])
                if len(raw) > 30 * 1024 * 1024:
                    raise ValueError("That file is larger than 30 MB. Please upload a sample "
                                     "of the data (the comparison subsamples big data anyway).")
                csv_text, notes = engine.any_to_csv(raw, b.get("name", ""))
                self._json(200, {"ok": True, "csv": csv_text, "notes": notes})
            else:
                self._send(404, "not found", "text/plain")
        except KeyError as e:
            self._json(400, {"ok": False, "error": f"The request is missing a required field, or names a column that is not in the file: {e}."})
        except Exception as e:
            self._json(400, {"ok": False, "error": str(e)})


if __name__ == "__main__":
    print(f"Imbalance Fixer running at http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
