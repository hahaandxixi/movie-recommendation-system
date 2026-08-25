import gzip

from myidea.scripts import build_posters


class _FakeResponse:
    def __init__(self, content: bytes, encoding: str | None = None):
        self._content = content
        self.headers = {"Content-Encoding": encoding} if encoding else {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self._content


def test_urlopen_reads_and_decompresses_gzip(monkeypatch):
    expected = b"poster page"
    response = _FakeResponse(gzip.compress(expected), "gzip")
    monkeypatch.setattr(build_posters.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    assert build_posters._urlopen("https://example.test") == expected
