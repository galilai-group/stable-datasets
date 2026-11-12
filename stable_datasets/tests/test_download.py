"""Regression tests for :func:`stable_datasets.utils.download`.

These spin up a throwaway local HTTP server rather than mocking, so the behaviour
under test is the real requests/urllib3 transfer path.
"""

import gzip
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from stable_datasets.utils import download


PAYLOAD = b"col_a,col_b\n" + b"".join(f"row{i},{i}\n".encode() for i in range(5000))


class _Handler(BaseHTTPRequestHandler):
    """Serves /plain verbatim and /gzipped with Content-Encoding: gzip."""

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        if self.path == "/gzipped":
            body = gzip.compress(PAYLOAD)
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
        else:
            body = PAYLOAD
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "text/csv")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_download_plain_response(http_server, tmp_path):
    path = download(f"{http_server}/plain", tmp_path, progress_bar=False, filename="plain.csv")
    assert path.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("*.tmp"))


def test_download_gzip_encoded_response(http_server, tmp_path):
    """Content-Length counts encoded bytes while requests writes decoded ones.

    Comparing the two rejected perfectly good downloads as "incomplete" -- this
    is what any server sending Content-Encoding: gzip (GitHub raw, for one) hits.
    """
    path = download(f"{http_server}/gzipped", tmp_path, progress_bar=False, filename="gzipped.csv")

    assert path.read_bytes() == PAYLOAD, "Body must be stored decoded and intact."
    assert path.stat().st_size == len(PAYLOAD)
    assert path.stat().st_size > len(gzip.compress(PAYLOAD)), "Sanity: decoded body is larger than the wire payload."
    assert not list(tmp_path.glob("*.tmp")), "A successful download must not leave a .tmp behind."


def test_download_gzip_encoded_response_is_cached(http_server, tmp_path):
    first = download(f"{http_server}/gzipped", tmp_path, progress_bar=False, filename="gzipped.csv")
    second = download(f"{http_server}/gzipped", tmp_path, progress_bar=False, filename="gzipped.csv")
    assert first == second
    assert second.read_bytes() == PAYLOAD


def test_download_gzip_encoded_response_verifies_checksum(http_server, tmp_path):
    """Checksums are computed over the decoded bytes, so they still apply."""
    import hashlib

    digest = hashlib.sha256(PAYLOAD).hexdigest()
    path = download(
        f"{http_server}/gzipped",
        tmp_path,
        progress_bar=False,
        filename="gzipped.csv",
        checksum=f"sha256:{digest}",
    )
    assert path.read_bytes() == PAYLOAD


class _IdentityAwareHandler(BaseHTTPRequestHandler):
    """Honours ``Accept-Encoding: identity``, the way GitHub raw does."""

    def do_GET(self):  # noqa: N802
        if "identity" in (self.headers.get("Accept-Encoding") or ""):
            body, encoding = PAYLOAD, None
        else:
            body, encoding = gzip.compress(PAYLOAD), "gzip"
        self.send_response(200)
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _EncodedRangeHandler(BaseHTTPRequestHandler):
    """Ignores Accept-Encoding and serves Range over the *encoded* stream."""

    def do_GET(self):  # noqa: N802
        body = gzip.compress(PAYLOAD)
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.split("=")[1].split("-")[0])
            chunk = body[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _plant_stale_tmp(directory, url, filename):
    """Write the .tmp an interrupted earlier attempt would have left behind."""
    import hashlib

    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    tmp = directory / f"{filename}.{digest}.tmp"
    tmp.write_bytes(PAYLOAD[:100])
    return tmp


def test_resume_asks_for_an_unencoded_body(tmp_path):
    """Range offsets index the encoded stream, the partial file holds decoded bytes.

    Requesting ``Accept-Encoding: identity`` on resume keeps the two consistent.
    """
    server = _serve(_IdentityAwareHandler)
    try:
        url = f"http://127.0.0.1:{server.server_port}/data"
        _plant_stale_tmp(tmp_path, url, "data")

        path = download(url, tmp_path, progress_bar=False, filename="data")
        assert path.read_bytes() == PAYLOAD
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        server.shutdown()
        server.server_close()


def test_undecodable_partial_download_does_not_wedge(tmp_path):
    """A stale .tmp plus an encoded range response must not poison every retry.

    Before the fix the decode error left the .tmp in place, so every subsequent
    attempt failed identically and the download could never recover.
    """
    server = _serve(_EncodedRangeHandler)
    try:
        url = f"http://127.0.0.1:{server.server_port}/data"
        _plant_stale_tmp(tmp_path, url, "data")

        with pytest.raises(RuntimeError):
            download(url, tmp_path, progress_bar=False, filename="data")

        assert not list(tmp_path.glob("*.tmp")), "Undecodable partial must be discarded."

        # A clean slate means the retry now succeeds.
        path = download(url, tmp_path, progress_bar=False, filename="data")
        assert path.read_bytes() == PAYLOAD
    finally:
        server.shutdown()
        server.server_close()


class _RangeHandler(BaseHTTPRequestHandler):
    """Supports Range properly, answering 416 when the offset is past the end."""

    def do_GET(self):  # noqa: N802
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.split("=")[1].split("-")[0])
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(PAYLOAD)}")
                self.end_headers()
                return
            chunk = PAYLOAD[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def log_message(self, *args):
        pass


def test_complete_but_unrenamed_partial_recovers(tmp_path):
    """A .tmp holding the whole file makes Range unsatisfiable.

    This is what a process killed between the final chunk write and the atomic
    rename leaves behind; the 416 must be recovered from, not wedge the download.
    """
    server = _serve(_RangeHandler)
    try:
        url = f"http://127.0.0.1:{server.server_port}/data"
        import hashlib

        digest = hashlib.sha256(url.encode()).hexdigest()[:10]
        (tmp_path / f"data.{digest}.tmp").write_bytes(PAYLOAD)

        path = download(url, tmp_path, progress_bar=False, filename="data")
        assert path.read_bytes() == PAYLOAD
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        server.shutdown()
        server.server_close()


def test_genuine_partial_download_still_resumes(tmp_path):
    """The 416 recovery must not turn every resume into a restart."""
    server = _serve(_RangeHandler)
    try:
        url = f"http://127.0.0.1:{server.server_port}/data"
        import hashlib

        digest = hashlib.sha256(url.encode()).hexdigest()[:10]
        (tmp_path / f"data.{digest}.tmp").write_bytes(PAYLOAD[:500])

        path = download(url, tmp_path, progress_bar=False, filename="data")
        assert path.read_bytes() == PAYLOAD, "Resumed bytes must reassemble the original payload."
    finally:
        server.shutdown()
        server.server_close()


def test_download_rejects_a_bad_checksum(http_server, tmp_path):
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(
            f"{http_server}/plain",
            tmp_path,
            progress_bar=False,
            filename="plain.csv",
            checksum="sha256:" + "0" * 64,
        )
