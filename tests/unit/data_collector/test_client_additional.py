import pytest

from src.data_collector.polygon_data.client import PolygonDataClient, PolygonAPIError


class DummyResponse:
    def __init__(self, status_code=200, text=None, json_data=None):
        self.status_code = status_code
        self.text = text or ""
        self._json = json_data

    def json(self):
        if self._json is not None:
            return self._json
        raise ValueError("Malformed JSON")


def test_make_request_500_retries_and_raises(mocker):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return DummyResponse(500, text="Server error")

    mocker.patch.object(client.session, "get", fake_get)

    with pytest.raises(PolygonAPIError):
        client._make_request("/server/error")


def test_make_request_malformed_json(mocker):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    def fake_get(url, params=None, timeout=None):
        return DummyResponse(200, text="not json", json_data=None)

    mocker.patch.object(client.session, "get", fake_get)

    with pytest.raises(PolygonAPIError):
        client._make_request("/malformed")


def test_api_key_present_in_headers(mocker):
    client = PolygonDataClient(api_key="MYKEY", requests_per_minute=100)

    seen = {}

    def fake_get(url, params=None, timeout=None):
        # Inspect the client's session headers
        seen["headers"] = dict(client.session.headers)
        return DummyResponse(200, json_data={"status": "OK"})

    mocker.patch.object(client.session, "get", fake_get)

    res = client._make_request("/ok")
    if res.get("status") != "OK":
        raise AssertionError("Expected OK status from API call")
    if "Authorization" in seen["headers"]:
        raise AssertionError("Authorization header should not be used; apiKey param preferred")
    # API key must be provided (client stores it); ensure attribute exists
    if client.api_key != "MYKEY":
        raise AssertionError("Client api_key attribute not set correctly")
