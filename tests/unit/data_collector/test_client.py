import pytest
import requests

from src.data_collector.polygon_data.client import PolygonDataClient, PolygonAPIError


class DummyResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


def test_make_request_success(mocker):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    # mocker session.get to return a successful response
    def fake_get(url, params=None, timeout=None):
        return DummyResponse(200, {"results": [1, 2, 3]})

    mocker.patch.object(client.session, "get", fake_get)

    data = client._make_request("/test/endpoint")
    assert "results" in data


def test_make_request_401_raises(mocker):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    def fake_get(url, params=None, timeout=None):
        return DummyResponse(401, {})

    mocker.patch.object(client.session, "get", fake_get)

    with pytest.raises(PolygonAPIError):
        client._make_request("/private")


def test_make_request_rate_limit_retries(mocker):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        # First two calls return 429, third call returns 200
        if calls["n"] < 3:
            return DummyResponse(429, {})
        return DummyResponse(200, {"status": "OK", "results": []})

    mocker.patch.object(client.session, "get", fake_get)

    data = client._make_request("/test/rate")
    assert data.get("status") == "OK"


def test_make_request_timeout_then_fail(mocker):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    def fake_get(url, params=None, timeout=None):
        raise requests.exceptions.Timeout()

    mocker.patch.object(client.session, "get", fake_get)

    with pytest.raises(PolygonAPIError):
        client._make_request("/timeout")


