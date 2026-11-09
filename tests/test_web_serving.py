"""Behaviour tests for serving the workbench files and the classic page from ``backend/api.py``.

``/`` and ``/index.html`` serve ``index.html`` of the web folder, every other plain file of that folder is served at
its own root path from an allowlist built at startup, ``/classic`` serves ``frontend/index.html``. The folder is
``FYD_WEB_DIR`` in these tests (a private temporary folder with known files, including hostile ones), plus one test that
checks the real package folder. The traversal tests use a real server and raw sockets, so the path the server sees is
exactly the text sent (an HTTP client library would tidy it first).

Run from the repository root::

    python -B -m unittest discover -s tests -p "test_web_serving.py" -v
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import types
import unittest
from unittest import mock

import api_support as support
from api_support import ServerProcess, processes_with, private_directory, remove_directory

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from jobs import Coordinator, JobConfig  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
FILES = {
    "index.html": b"<!doctype html><title>workbench</title><script src=\"app.js\"></script>\n",
    "app.js": b"console.log('app');\n",
    "style.css": b"body{margin:0}\n",
    "data.json": b'{"a": 1}\n',
    "pic.png": PNG,
    "icon.svg": b"<svg xmlns='http://www.w3.org/2000/svg'/>",
    "report.html": b"<!doctype html><title>report</title><p>report</p>\n",
    "notes.txt": b"plain notes\n",
    ".hidden.html": b"<p>hidden</p>",
    ".nojekyll": b"",
    "notes.md": b"markdown is not in the content type table",
    "no_extension": b"x",
    "bad name.html": b"<p>space in the name</p>",
    "UPPER.JS": b"console.log('upper');\n",
}
SERVED = ("index.html", "app.js", "style.css", "data.json", "pic.png", "icon.svg", "report.html", "notes.txt", "UPPER.JS")
NOT_SERVED = (".hidden.html", ".nojekyll", "notes.md", "no_extension", "bad name.html")
CONTENT_TYPES = {"app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8",
                 "data.json": "application/json", "pic.png": "image/png", "icon.svg": "image/svg+xml",
                 "report.html": "text/html; charset=utf-8", "notes.txt": "text/plain; charset=utf-8"}
SECRET = b"OUTSIDE-THE-WEB-FOLDER-SECRET"


def make_web_folder(base: Path) -> Path:
    web = base / "web"
    web.mkdir()
    for name, content in FILES.items():
        (web / name).write_bytes(content)
    (web / "sub").mkdir()
    (web / "sub" / "secret.txt").write_bytes(b"inside a subfolder")
    (web / "sub" / "deep.js").write_bytes(b"x")
    (base / "outside.txt").write_bytes(SECRET)
    return web


class FakeTrack(types.SimpleNamespace):
    """Stands in for ``track`` so no usage database is touched; records every event."""

    def __init__(self):
        super().__init__(calls=[], STATS_KEY="")

    def log_event(self, kind, detail="", ip="", user_agent="", referrer="", is_owner=0):
        self.calls.append((kind, detail, user_agent))

    def owner_token(self):
        return "owner-token-for-this-test"


class ServingCase(unittest.TestCase):
    """A TestClient app over a private web folder; ``track`` is replaced by a recorder."""

    @classmethod
    def setUpClass(cls):
        cls.base = private_directory("fyd-web-test-")
        cls.web = make_web_folder(cls.base)
        cls.track = FakeTrack()
        cls._patch = mock.patch.object(api, "track_module", lambda: cls.track)
        cls._patch.start()
        cls.start_app()

    @classmethod
    def start_app(cls, web: Path | None = None):
        with mock.patch.dict(os.environ, {"FYD_WEB_DIR": str(web or cls.web)}):
            cls.coordinator = Coordinator(JobConfig(temp_root=str(cls.base / "root"), sweep_interval_seconds=0.5))
            cls.app = api.create_app(coordinator=cls.coordinator)
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.client.__enter__()

    @classmethod
    def stop_app(cls):
        cls.client.__exit__(None, None, None)

    @classmethod
    def tearDownClass(cls):
        cls.stop_app()
        cls._patch.stop()
        problems = []
        if cls.coordinator.store.root.exists():
            problems.append("the storage root was not deleted")
        remove_directory(cls.base)
        if cls.base.exists():
            problems.append("the private directory could not be removed")
        if problems:
            raise AssertionError("; ".join(problems))

    def setUp(self):
        self.track.calls.clear()


class ServingTests(ServingCase):
    def test_root_and_index_html_serve_the_workbench_index_and_log_a_visit(self):
        for path in ("/", "/index.html"):
            self.track.calls.clear()
            response = self.client.get(path, headers={"User-Agent": "tester"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, FILES["index.html"])
            self.assertEqual(response.headers["content-type"], "text/html; charset=utf-8")
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
            self.assertEqual(response.headers["x-frame-options"], "DENY")
            self.assertEqual(response.headers["referrer-policy"], "no-referrer")
            self.assertIn("x-request-id", response.headers)
            self.assertEqual(self.track.calls, [("visit", "", "tester")])   # the same event the old page logged

    def test_head_is_like_get_without_a_body_and_is_not_a_visit(self):
        for name in ("/", "/index.html", "/app.js", "/data.json", "/pic.png", "/report.html"):
            file_name = "index.html" if name in ("/", "/index.html") else name[1:]
            got, head = self.client.get(name), self.client.head(name)
            self.assertEqual(head.status_code, 200, name)
            self.assertEqual(head.content, b"", name)
            self.assertEqual(head.headers["content-length"], str(len(FILES[file_name])), name)
            for header in ("content-type", "cache-control", "x-content-type-options"):
                self.assertEqual(head.headers[header], got.headers[header], (name, header))
        self.assertEqual(len(self.track.calls), 2)   # only the two GET requests for the index page were visits

    def test_every_other_plain_file_is_served_at_its_root_path_with_a_fixed_content_type(self):
        for name in SERVED:
            if name == "index.html":
                continue
            response = self.client.get("/" + name)
            self.assertEqual(response.status_code, 200, name)
            self.assertEqual(response.content, FILES[name], name)
            expected = CONTENT_TYPES.get(name, "text/javascript; charset=utf-8")
            self.assertEqual(response.headers["content-type"], expected, name)
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
            if name == "report.html":
                self.assertEqual(response.headers["cache-control"], "no-store")   # a page is never cached
            else:
                self.assertEqual(response.headers["cache-control"], "public, max-age=300", name)
        self.assertEqual(self.track.calls, [])   # assets are not visits

    def test_report_pages_and_svg_carry_the_document_policy(self):
        for name in ("/report.html", "/icon.svg"):
            for method in ("get", "head"):
                policy = getattr(self.client, method)(name).headers["content-security-policy"]
                self.assertEqual(policy, api.DOCUMENT_CSP, name)
                self.assertIn("default-src 'none'", policy)
                self.assertIn("sandbox", policy)

    def test_the_workbench_index_has_its_own_policy_on_get_and_head_and_nothing_else_does(self):
        expected = {"default-src": "'none'", "script-src": "'self'", "style-src": "'self'", "img-src": "'self' data:",
                    "connect-src": "'self'", "font-src": "'self'", "base-uri": "'none'", "form-action": "'none'",
                    "frame-ancestors": "'none'"}
        for path in ("/", "/index.html"):
            for method in ("get", "head"):
                policy = getattr(self.client, method)(path).headers["content-security-policy"]
                parsed = dict(item.strip().split(" ", 1) for item in policy.split(";") if item.strip())
                self.assertEqual(parsed, expected, (method, path))
                self.assertNotIn("unsafe", policy)
                self.assertNotIn("sandbox", policy)
        # scripts, styles, data and other files carry no policy of their own
        for name in ("/app.js", "/style.css", "/data.json", "/pic.png", "/notes.txt"):
            self.assertNotIn("content-security-policy", self.client.get(name).headers, name)
        # the old page uses inline script, so it must have none
        for method in ("get", "head"):
            self.assertNotIn("content-security-policy", getattr(self.client, method)("/classic").headers)

    def test_hidden_odd_and_unlisted_names_are_not_served(self):
        for name in NOT_SERVED:
            for target in ("/" + name, "/" + name.replace(" ", "%20")):
                response = self.client.get(target)
                self.assertEqual((response.status_code, response.text), (404, "not found"), target)
        for target in ("/sub", "/sub/", "/sub/secret.txt", "/secret.txt", "/deep.js", "/sub/deep.js", "/web", "/root"):
            self.assertEqual(self.client.get(target).status_code, 404, target)

    def test_case_and_trailing_variants_do_not_match(self):
        for target in ("/APP.JS", "/App.js", "/app.js/", "/INDEX.HTML", "/index.html/", "/upper.js"):
            self.assertEqual(self.client.get(target).status_code, 404, target)
        self.assertEqual(self.client.get("/UPPER.JS").status_code, 200)

    def test_other_methods_on_a_served_name_are_the_legacy_404(self):
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            response = self.client.request(method, "/app.js", headers={"Origin": "http://testserver"})
            self.assertEqual((response.status_code, response.text), (404, "not found"), method)
            self.assertEqual(self.client.request(method, "/").status_code, 404, method)

    def test_a_query_string_does_not_choose_a_file(self):
        response = self.client.get("/app.js?name=../outside.txt")
        self.assertEqual(response.content, FILES["app.js"])
        self.assertNotEqual(self.client.get("/?file=outside.txt").content, SECRET)

    def test_the_allowlist_is_built_once_at_startup(self):
        late = self.web / "late.js"
        late.write_bytes(b"late")
        self.addCleanup(late.unlink)
        self.assertEqual(self.client.get("/late.js").status_code, 404)   # created after startup: not served
        gone = self.web / "gone.js"
        gone.write_bytes(b"gone")
        self.stop_app()
        type(self).start_app()
        self.assertEqual(self.client.get("/gone.js").status_code, 200)   # present at startup
        gone.unlink()
        self.assertEqual(self.client.get("/gone.js").status_code, 404)   # removed afterwards: a clean 404, no error

    def test_a_symbolic_link_is_not_served(self):
        link = self.web / "link.js"
        try:
            os.symlink(self.base / "outside.txt", link)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links cannot be created here")
        self.addCleanup(link.unlink)
        self.stop_app()
        type(self).start_app()
        self.assertEqual(self.client.get("/link.js").status_code, 404)
        self.assertNotIn("link.js", self.app.state.web_assets)

    def test_a_file_over_the_size_bound_is_not_served(self):
        with mock.patch.object(api, "MAX_WEB_FILE_BYTES", 15):
            assets = api.build_web_assets(self.web)
        self.assertEqual(sorted(assets), ["data.json", "notes.txt", "style.css"])   # 9, 12 and 15 bytes; the rest is larger
        self.assertGreaterEqual(api.MAX_WEB_FILE_BYTES, 1024 * 1024)   # the real bound leaves room for real files

    def test_the_allowlist_holds_only_plain_regular_files(self):
        self.assertEqual(sorted(self.app.state.web_assets), sorted(SERVED))
        for name, (path, content_type) in self.app.state.web_assets.items():
            self.assertTrue(path.is_file() and not path.is_symlink())
            self.assertRegex(name, r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    def test_classic_serves_the_old_page_byte_for_byte(self):
        old = (support.ROOT / "frontend" / "index.html").read_bytes()
        response = self.client.get("/classic", headers={"User-Agent": "tester"})
        self.assertEqual(response.content, old)
        self.assertEqual(response.headers["content-type"], "text/html; charset=utf-8")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("referrer-policy", response.headers)
        self.assertEqual(self.track.calls, [("visit", "", "tester")])
        self.assertEqual(self.client.get("/classic/").status_code, 404)
        self.assertEqual(self.client.get("/classic.html").status_code, 404)

    def test_the_visit_event_of_the_workbench_and_of_classic_is_identical(self):
        self.client.get("/", headers={"User-Agent": "same"})
        self.client.get("/classic", headers={"User-Agent": "same"})
        self.assertEqual(self.track.calls[0], self.track.calls[1])

    def test_existing_compatibility_routes_are_unchanged(self):
        self.assertEqual(self.client.get("/healthz").text, "ok")
        self.assertEqual(self.client.get("/api/v1/version").status_code, 200)
        self.assertEqual(self.client.get("/api/samples").status_code, 200)
        self.assertEqual(self.client.get("/api/sample?name=liver_disease.csv").headers["content-type"],
                         "text/csv")
        self.assertEqual(self.client.get("/stats").text, "not found")
        self.assertEqual(self.client.get("/definitely-not-a-page").text, "not found")
        self.assertEqual(self.client.get("/api/v1/definitely-not-a-route").status_code, 404)


class MissingIndexTests(unittest.TestCase):
    def test_without_an_index_the_root_says_so_and_the_other_files_and_classic_still_work(self):
        base = private_directory("fyd-web-noindex-")
        try:
            web = base / "web"
            web.mkdir()
            (web / "app.js").write_bytes(b"x")
            with mock.patch.dict(os.environ, {"FYD_WEB_DIR": str(web)}):
                coordinator = Coordinator(JobConfig(temp_root=str(base / "root")))
                app = api.create_app(coordinator=coordinator)
            with TestClient(app, raise_server_exceptions=False) as client, \
                    mock.patch.object(api, "track_module", lambda: FakeTrack()):
                for path in ("/", "/index.html"):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 503)
                    self.assertIn("not installed", response.text)
                self.assertEqual(client.get("/app.js").status_code, 200)
                self.assertEqual(client.get("/classic").status_code, 200)
            missing = base / "nowhere"
            with mock.patch.dict(os.environ, {"FYD_WEB_DIR": str(missing)}):
                self.assertEqual(api.build_web_assets(api.web_directory()), {})
        finally:
            remove_directory(base)

    def test_the_default_folder_is_the_package_web_folder(self):
        self.assertEqual(api.DEFAULT_WEB_DIR, support.ROOT / "evaluation" / "src" / "rare_event_eval" / "web")
        with mock.patch.dict(os.environ, {"FYD_WEB_DIR": ""}):
            self.assertEqual(api.web_directory(), api.DEFAULT_WEB_DIR)


class RealWebFolderTests(unittest.TestCase):
    """The package's own folder: every plain file with a known type is served with its exact bytes."""

    @classmethod
    def setUpClass(cls):
        cls.base = private_directory("fyd-web-real-")
        with mock.patch.dict(os.environ, {"FYD_WEB_DIR": ""}):
            cls.coordinator = Coordinator(JobConfig(temp_root=str(cls.base / "root")))
            cls.app = api.create_app(coordinator=cls.coordinator)
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        remove_directory(cls.base)

    def test_the_index_and_assets_are_the_package_files(self):
        web = api.DEFAULT_WEB_DIR
        self.assertTrue((web / "index.html").is_file())
        index = (web / "index.html").read_bytes()
        with mock.patch.object(api, "track_module", lambda: FakeTrack()):
            self.assertEqual(self.client.get("/").content, index)
            self.assertEqual(self.client.get("/index.html").content, index)
        served = 0
        for path in sorted(web.iterdir()):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in api.WEB_CONTENT_TYPES:
                response = self.client.get("/" + path.name)
                self.assertEqual(response.status_code, 200, path.name)
                self.assertEqual(response.content, path.read_bytes(), path.name)
                served += 1
        self.assertGreaterEqual(served, 5)   # index, scripts, style and the preview image at least
        self.assertEqual(self.client.get("/.nojekyll").status_code, 404)
        self.assertEqual(set(self.app.state.web_assets), {path.name for path in web.iterdir() if path.is_file()
                         and not path.name.startswith(".") and path.suffix.lower() in api.WEB_CONTENT_TYPES})


def raw_request(port: int, request_line: str, method_headers: str = "") -> tuple:
    """Send one HTTP/1.1 request with the path exactly as given. Returns ``(status, headers text, body bytes)``."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(f"{request_line}\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n{method_headers}\r\n".encode("latin-1"))
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    data = b"".join(chunks)
    head, _, body = data.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1]) if head.startswith(b"HTTP/") else 0
    return status, head.decode("latin-1"), body


class TraversalAndHeadOnARealServer(unittest.TestCase):
    """Hostile paths and HEAD against ``python backend/api.py``. Only paths that are not visits are requested, so no
    usage database is opened."""

    @classmethod
    def setUpClass(cls):
        cls.base = private_directory("fyd-web-server-")
        cls.web = make_web_folder(cls.base)
        cls.server = ServerProcess(cls.base, {"FYD_WEB_DIR": str(cls.web)})
        cls.marker = cls.base.name
        cls.server.wait_ready()

    @classmethod
    def tearDownClass(cls):
        code = cls.server.stop()
        problems = []
        if not cls.server.stopped_cleanly(code):
            problems.append(f"the server did not stop cleanly (exit {code})")
        leftover = processes_with(cls.marker)
        if leftover:
            for pid in leftover:
                support.kill_pid(pid)
            problems.append("a process was left running")
        remove_directory(cls.base)
        if problems:
            raise AssertionError("; ".join(problems))

    def test_hostile_paths_are_404_and_never_leak_a_file_outside_the_folder(self):
        paths = [
            "/..%2f..%2foutside.txt", "/..%2foutside.txt", "/%2e%2e/outside.txt", "/%2e%2e%2foutside.txt",
            "/%2E%2E%2Foutside.txt", "/..%5coutside.txt", "/..%252foutside.txt", "/static//", "/static/../outside.txt",
            "//outside.txt", "///app.js", "/app.js/../outside.txt", "/.", "/..", "/%2e", "/%2e%2e",
            "/sub", "/sub/", "/sub/secret.txt", "/.hidden.html", "/%2ehidden.html", "/.nojekyll", "/app.js%00.txt",
            "/app.js%00", "/app.js%20", "/app.js%2f", "/index.html%2f", "/web", "/web/", "/app.js;x", "/..%2fbackend%2fapi.py",
            "/..%2f..%2fbackend%2fapi.py", "/%2e%2e/%2e%2e/backend/api.py", "/notes.md", "/no_extension",
            "/bad%20name.html", "/C:%5cwindows%5cwin.ini", "/%c0%ae%c0%ae/outside.txt", "/outside.txt",
        ]
        for path in paths:
            status, head, body = raw_request(self.server.port, f"GET {path} HTTP/1.1")
            self.assertNotIn(SECRET, body, path)
            self.assertNotIn(b"inside a subfolder", body, path)
            self.assertNotIn(b"<p>hidden</p>", body, path)
            self.assertNotIn(b"def create_app", body, path)   # nothing from backend/api.py
            self.assertNotIn(str(self.base).encode(), body, path)
            self.assertIn(status, (400, 404), (path, status))
        # "/./app.js" may be collapsed to "/app.js" by a server; either answer is fine, a leak is not
        status, _, body = raw_request(self.server.port, "GET /./app.js HTTP/1.1")
        self.assertIn(status, (200, 404))
        self.assertNotIn(SECRET, body)

    def test_a_plain_name_is_served_and_head_has_the_same_headers_and_no_body(self):
        status, head, body = raw_request(self.server.port, "GET /app.js HTTP/1.1")
        self.assertEqual((status, body), (200, FILES["app.js"]))
        self.assertIn("content-type: text/javascript; charset=utf-8", head.lower())
        self.assertIn("cache-control: public, max-age=300", head.lower())
        self.assertIn("x-content-type-options: nosniff", head.lower())
        for name in ("/app.js", "/", "/index.html", "/report.html"):
            file_name = "index.html" if name in ("/", "/index.html") else name[1:]
            status, head, body = raw_request(self.server.port, f"HEAD {name} HTTP/1.1")
            self.assertEqual((status, body), (200, b""), name)
            self.assertIn(f"content-length: {len(FILES[file_name])}", head.lower(), name)
        status, _, _ = raw_request(self.server.port, "HEAD /sub/secret.txt HTTP/1.1")
        self.assertEqual(status, 404)

    def test_the_server_still_answers_after_the_hostile_requests(self):
        status, _, body = raw_request(self.server.port, "GET /healthz HTTP/1.1")
        self.assertEqual((status, body), (200, b"ok"))


if __name__ == "__main__":
    unittest.main()
