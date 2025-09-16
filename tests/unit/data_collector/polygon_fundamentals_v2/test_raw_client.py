import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from src.data_collector.polygon_fundamentals_v2.raw_client import (
    RawPolygonFundamentalsClient,
)
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
    """
    Run the given coroutine to completion using asyncio.run and return its result.
    
    Parameters:
        coro: An awaitable or coroutine object to execute.
    
    Returns:
        The value returned by the coroutine.
    
    Notes:
        - Executes the coroutine in a new event loop; calling this when an asyncio event loop is already running will raise a RuntimeError.
        - Any exception raised by the coroutine is propagated to the caller.
    """
    return asyncio.run(coro)


def test_get_financials_raw_success():
    """Successfully fetch raw financials and return parsed JSON payload"""
    # Setup
    pf_config.polygon_fundamentals_config.API_KEY = "testkey"
    payload = {"status": "OK", "results": []}

    fake_session = _FakeSession([_FakeResp(200, payload)])

    with patch("aiohttp.ClientSession", lambda *a, **k: fake_session), patch(
        "src.data_collector.polygon_fundamentals_v2.raw_client.BasicRateLimiter",
        _FakeRateLimiter,
    ):

        # Execution
        async def _call():
            """
            Fetch raw financials for the ticker "TST" using RawPolygonFundamentalsClient and return the parsed JSON response.
            
            Returns:
                dict: JSON payload returned by get_financials_raw for "TST".
            """
            async with RawPolygonFundamentalsClient() as client:
                return await client.get_financials_raw("TST")

        res = _run(_call())

    # Verification
    assert isinstance(res, dict)
    assert res["status"] == "OK"


def test_get_financials_raw_rate_limit_retry():
    """Retry on 429 responses and succeed when service returns 200"""
    # Setup: first two responses 429 then 200
    pf_config.polygon_fundamentals_config.API_KEY = "testkey"
    payload_ok = {"status": "OK"}

    fake_session = _FakeSession(
        [
            _FakeResp(429, {}),
            _FakeResp(429, {}),
            _FakeResp(200, payload_ok),
        ]
    )

    with patch("aiohttp.ClientSession", lambda *a, **k: fake_session), patch(
        "src.data_collector.polygon_fundamentals_v2.raw_client.BasicRateLimiter",
        _FakeRateLimiter,
    ), patch("asyncio.sleep", new=AsyncMock(return_value=None)):

        # Execution
        async def _call():
            """
            Fetch raw financials for the ticker "TST" using RawPolygonFundamentalsClient and return the parsed JSON response.
            
            Returns:
                dict: JSON payload returned by get_financials_raw for "TST".
            """
            async with RawPolygonFundamentalsClient() as client:
                return await client.get_financials_raw("TST")

        res = _run(_call())

    # Verification
    assert res["status"] == "OK"

def test_get_financials_raw_raises_on_no_session():
    """Internal _get should raise when no aiohttp session is available"""
    client = RawPolygonFundamentalsClient()
    with pytest.raises(RuntimeError):
        _run(client._get("http://example.com", {}))