import pytest
import requests

from src.data_collector.polygon_data.client import PolygonDataClient, PolygonAPIError


def test_make_request_success(mocker, fake_response_factory):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    resp = fake_response_factory(status=200, json_data={"results": [1, 2, 3]})
    mocker.patch.object(client.session, "get", return_value=resp)

    data = client._make_request("/test/endpoint")
    if "results" not in data:
        raise AssertionError("Expected 'results' key in API response data")


def test_make_request_401_raises(mocker, fake_response_factory):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    resp = fake_response_factory(status=401, json_data={})
    mocker.patch.object(client.session, "get", return_value=resp)

    with pytest.raises(PolygonAPIError):
        client._make_request("/private")


def test_make_request_rate_limit_retries(mocker, fake_response_factory):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        # First two calls return 429, third call returns 200
        if calls["n"] < 3:
            return fake_response_factory(status=429, json_data={})
        return fake_response_factory(status=200, json_data={"status": "OK", "results": []})

    mocker.patch.object(client.session, "get", side_effect=fake_get)

    data = client._make_request("/test/rate")
    if data.get("status") != "OK":
        raise AssertionError("Rate-limited request did not eventually return OK status")


def test_make_request_timeout_then_fail(mocker):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    def fake_get(url, params=None, timeout=None):
        raise requests.exceptions.Timeout()

    mocker.patch.object(client.session, "get", side_effect=fake_get)

    with pytest.raises(PolygonAPIError):
        client._make_request("/timeout")
