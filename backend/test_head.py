"""HEAD must mirror GET (same status and headers, empty body) and must not count as a visit.
Run: py -3.11 backend/test_head.py"""
import http.client
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer

sys.path.insert(0, "backend")
import server  # noqa: E402


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.visits = []
        cls._orig = server.track.log_event
        server.track.log_event = lambda kind, *a, **k: cls.visits.append(kind)

    @classmethod
    def tearDownClass(cls):
        server.track.log_event = cls._orig
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _req(self, method, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path)
        r = c.getresponse()
        out = (r.status, dict(r.getheaders()), r.read())
        c.close()
        return out

    def test_head_matches_get_without_body(self):
        for path in ("/", "/healthz", "/api/samples", "/does-not-exist"):
            gs, gh, gb = self._req("GET", path)
            hs, hh, hb = self._req("HEAD", path)
            self.assertEqual(hs, gs, path)
            self.assertEqual(hh["Content-Type"], gh["Content-Type"], path)
            self.assertEqual(hh["Content-Length"], str(len(gb)), path)
            self.assertEqual(hb, b"", path)

    def test_head_is_not_logged_as_a_visit(self):
        self.visits.clear()
        self._req("HEAD", "/")
        self.assertEqual(self.visits, [])
        self._req("GET", "/")
        self.assertEqual(self.visits, ["visit"])


if __name__ == "__main__":
    unittest.main()
