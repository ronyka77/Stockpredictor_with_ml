import asyncio
import pytest

from src.data_collector.polygon_fundamentals_v2.raw_client import RawPolygonFundamentalsClient


class DummySession:
    def __init__(self, responses):
        self._responses = responses

    async def get(self, url, params=None):
        class RespCtx:
            def __init__(self, status, payload):
                self.status = status
                self._payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self):
                return self._payload

            async def text(self):
                return str(self._payload)

        # pop one response
        status, payload = self._responses.pop(0)
        return RespCtx(status, payload)


# def test_raw_client_retries_on_429(monkeypatch):
#     cfg = None
#     client = RawPolygonFundamentalsClient(cfg)

#     # inject dummy session and rate limiter
#     client.session = DummySession([(429, {"error": "rate"}), (200, {"ok": True})])

#     res = asyncio.run(client._get("http://example", {}))
#     assert res == {"ok": True}


