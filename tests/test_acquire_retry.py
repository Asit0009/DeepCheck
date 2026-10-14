import pytest
import requests

from src.data import acquire


class FakeResponse:
    def __init__(self, status=200, body='{"shards": 3}'):
        self.status_code, self._body = status, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        import json
        return json.loads(self._body)         # raises JSONDecodeError (a ValueError) on an empty / HTML body


@pytest.fixture
def scripted(monkeypatch):
    """Make requests.get return the scripted replies in order; never really sleep."""
    calls = {"n": 0, "sleeps": []}

    def install(replies):
        def fake_get(url, timeout=None):
            reply = replies[min(calls["n"], len(replies) - 1)]
            calls["n"] += 1
            if isinstance(reply, Exception):
                raise reply
            return reply
        monkeypatch.setattr(acquire.requests, "get", fake_get)
        monkeypatch.setattr(acquire.time, "sleep", lambda s: calls["sleeps"].append(s))
        return calls
    return install


def test_returns_json_on_first_success(scripted):
    calls = scripted([FakeResponse()])
    assert acquire.get_json("http://x") == {"shards": 3}
    assert calls["n"] == 1 and calls["sleeps"] == []


def test_recovers_from_empty_html_5xx_and_connection_errors(scripted):
    calls = scripted([FakeResponse(body=""), FakeResponse(body="<html>rate limited</html>"), FakeResponse(status=503),
                      requests.ConnectionError("reset"), FakeResponse()])
    assert acquire.get_json("http://x") == {"shards": 3}
    assert calls["n"] == 5 and len(calls["sleeps"]) == 4
    assert calls["sleeps"] == sorted(calls["sleeps"])          # backoff never shrinks


def test_gives_up_with_a_clear_error_after_the_retry_budget(scripted):
    calls = scripted([FakeResponse(body="")])
    with pytest.raises(IOError, match="could not fetch JSON from http://x after 3 tries"):
        acquire.get_json("http://x", retries=3)
    assert calls["n"] == 3 and len(calls["sleeps"]) == 2        # no pointless sleep after the last attempt
