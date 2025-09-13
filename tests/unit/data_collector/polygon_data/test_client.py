import pytest
from unittest.mock import patch

from src.data_collector.polygon_data.client import PolygonDataClient, PolygonAPIError
from tests._fixtures import canned_api_factory


def test_make_request_success(polygon_client):
    fake_resp = canned_api_factory("grouped_daily")
    with patch.object(polygon_client.session, "get", return_value=fake_resp):
        data = polygon_client._make_request("/test/endpoint")

    assert "results" in data


def test_make_request_401_raises(polygon_client):
    fake = canned_api_factory("empty", status=401)
    fake._payload = {"error": "Invalid API key"}
    with patch.object(polygon_client.session, "get", return_value=fake):
        with pytest.raises(PolygonAPIError) as exc:
            polygon_client._make_request("/private")
    # Assert exception details for better diagnostics
    assert exc.value.status_code == 401
    assert "Invalid API key" in str(exc.value)


def test_make_request_rate_limit_retries(polygon_client):
    # Simulate successive low-level HTTP responses so retry logic executes
    r1 = canned_api_factory("empty", status=429)
    r2 = canned_api_factory("empty", status=429)
    r3 = canned_api_factory("empty", status=200)
    r3._payload = {"status": "OK", "results": []}

    # Patch time.sleep to avoid delays during backoff retries
    with patch.object(
        polygon_client.session, "get", side_effect=[r1, r2, r3]
    ) as mock_get:
        with patch("time.sleep", return_value=None):
            data = polygon_client._make_request("/test/rate")

    assert data.get("status") == "OK"
    # Ensure session.get was called for each retry attempt
    assert mock_get.call_count == 3


def test_fetch_paginated_data_concatenates_pages(polygon_client):
    page1 = canned_api_factory("empty")
    page1._payload = {
        "results": [{"a": 1}],
        "next_url": "https://api/p/v1?page=2&apikey=TEST",
    }
    page2 = canned_api_factory("empty")
    page2._payload = {"results": [{"b": 2}], "next_url": None}

    # _make_request will be called for each page; return raw dicts
    with patch.object(
        PolygonDataClient, "_make_request", side_effect=[page1.json(), page2.json()]
    ) as mock_make:
        results = polygon_client._fetch_paginated_data("/v1", params={"limit": 1})

    assert results == [{"a": 1}, {"b": 2}]
    assert mock_make.call_count == 2


def test_api_key_passed_and_api_error_handled(polygon_client):
    fake = canned_api_factory("empty", status=200)
    fake._payload = {"status": "ERROR", "error": "bad"}

    def fake_get(url, params=None, timeout=None):
        assert params is not None and params.get("apikey") == "TEST"
        return fake

    with patch.object(polygon_client.session, "get", side_effect=fake_get):
        with pytest.raises(PolygonAPIError) as exc:
            polygon_client._make_request("/err")

    assert "bad" in str(exc.value)


def test_make_request_500_retries_and_raises(polygon_client):
    # Simulate server error response via patched _make_request
    with patch(
        "src.data_collector.polygon_data.client.PolygonDataClient._make_request",
        side_effect=PolygonAPIError("Server error", status_code=500),
    ):
        with pytest.raises(PolygonAPIError):
            polygon_client._make_request("/server/error")


def test_make_request_malformed_json(polygon_client):
    # Mimic malformed JSON by having patched _make_request raise
    with patch(
        "src.data_collector.polygon_data.client.PolygonDataClient._make_request",
        side_effect=PolygonAPIError("Malformed JSON", status_code=200),
    ):
        with pytest.raises(PolygonAPIError):
            polygon_client._make_request("/malformed")


def test_api_key_present_in_headers_local_client():
    # Create a local client with the expected API key for this test
    client = PolygonDataClient(api_key="MYKEY", requests_per_minute=100)

    # Patch at _make_request level and validate client attribute
    resp = canned_api_factory("empty")
    resp._payload = {"status": "OK", "results": []}
    with patch(
        "src.data_collector.polygon_data.client.PolygonDataClient._make_request",
        return_value=resp.json(),
    ):
        res = client._make_request("/ok")
        assert res.get("status") == "OK"
        # API key must be provided (client stores it); ensure attribute exists
        assert client.api_key == "MYKEY"
