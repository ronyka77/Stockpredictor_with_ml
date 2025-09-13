import asyncio
from unittest import mock

import pytest

from src.data_collector.polygon_fundamentals_v2.raw_client import RawPolygonFundamentalsClient
from src.data_collector.polygon_fundamentals import config as pf_config


class _FakeResp:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload


class _FakeCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, resp_sequence):
        # resp_sequence is an iterator yielding _FakeResp instances
        self._iter = iter(resp_sequence)

    def get(self, url, params=None):
        return _FakeCM(next(self._iter))

    async def close(self):
        return None


class _FakeRateLimiter:
    def __init__(self, requests_per_minute=None):
        # accept the same signature as the real rate limiter
        self.requests_per_minute = requests_per_minute

    async def wait_if_needed(self):
        return None


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_get_financials_raw_success(monkeypatch):
    # Setup
    pf_config.polygon_fundamentals_config.API_KEY = "testkey"
    payload = {"status": "OK", "results": []}

    fake_session = _FakeSession([_FakeResp(200, payload)])

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: fake_session)
    monkeypatch.setattr(
        "src.data_collector.polygon_fundamentals_v2.raw_client.BasicRateLimiter",
        _FakeRateLimiter,
    )

    # Execution
    async def _call():
        async with RawPolygonFundamentalsClient() as client:
            return await client.get_financials_raw("TST")

    res = _run(_call())

    # Verification
    assert isinstance(res, dict)
    assert res["status"] == "OK"


def test_get_financials_raw_rate_limit_retry(monkeypatch):
    # Setup: first two responses 429 then 200
    pf_config.polygon_fundamentals_config.API_KEY = "testkey"
    payload_ok = {"status": "OK"}

    fake_session = _FakeSession([
        _FakeResp(429, {}),
        _FakeResp(429, {}),
        _FakeResp(200, payload_ok),
    ])

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: fake_session)
    monkeypatch.setattr(
        "src.data_collector.polygon_fundamentals_v2.raw_client.BasicRateLimiter",
        _FakeRateLimiter,
    )

    # Prevent real asyncio.sleep delays
    async def _nosleep(*args, **kwargs):
        return None

    monkeypatch.setattr("asyncio.sleep", _nosleep)

    # Execution
    async def _call():
        async with RawPolygonFundamentalsClient() as client:
            return await client.get_financials_raw("TST")

    res = _run(_call())

    # Verification
    assert res["status"] == "OK"


