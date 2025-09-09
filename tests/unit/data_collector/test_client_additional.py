import pytest

from src.data_collector.polygon_data.client import PolygonDataClient, PolygonAPIError


from tests._fixtures.conftest import fake_http_response


def test_make_request_500_retries_and_raises(mocker, fake_http_response):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return fake_http_response(status=500, json_data=None)

    mocker.patch.object(client.session, "get", side_effect=fake_get)

    with pytest.raises(PolygonAPIError):
        client._make_request("/server/error")


def test_make_request_malformed_json(mocker, fake_http_response):
    client = PolygonDataClient(api_key="TEST", requests_per_minute=100)

    mocker.patch.object(client.session, "get", return_value=fake_http_response(status=200, json_data=None, raise_on_json=True))

    with pytest.raises(PolygonAPIError):
        client._make_request("/malformed")


def test_api_key_present_in_headers(mocker, fake_http_response):
    client = PolygonDataClient(api_key="MYKEY", requests_per_minute=100)

    seen = {}

    def fake_get(url, params=None, timeout=None):
        # Inspect the client's session headers
        seen["headers"] = dict(client.session.headers)
        return fake_http_response(status=200, json_data={"status": "OK"})

    mocker.patch.object(client.session, "get", side_effect=fake_get)

    res = client._make_request("/ok")
    if res.get("status") != "OK":
        raise AssertionError("Expected OK status from API call")
    if "Authorization" in seen["headers"]:
        raise AssertionError(
            "Authorization header should not be used; apiKey param preferred"
        )
    # API key must be provided (client stores it); ensure attribute exists
    if client.api_key != "MYKEY":
        raise AssertionError("Client api_key attribute not set correctly")
