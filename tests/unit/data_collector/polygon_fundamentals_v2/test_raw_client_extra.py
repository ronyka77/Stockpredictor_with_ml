import asyncio
import pytest

from src.data_collector.polygon_fundamentals_v2.raw_client import RawPolygonFundamentalsClient


class _TimeoutResp:
    def __init__(self):
        self.status = 200

    async def json(self):
        raise asyncio.TimeoutError()


class _FakeCMTimeout:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSessionTimeout:
    def get(self, url, params=None):
        return _FakeCMTimeout(_TimeoutResp())

    async def close(self):
        return None


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_get_financials_raw_raises_on_no_session(monkeypatch):
    # Ensure that calling _get without session raises as a guard (internal behavior)
    client = RawPolygonFundamentalsClient()
    with pytest.raises(RuntimeError):
        _run(client._get("http://example.com", {}))


