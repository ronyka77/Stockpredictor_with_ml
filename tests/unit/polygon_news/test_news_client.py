import pytest
from datetime import datetime, timezone

from src.data_collector.polygon_news.news_client import PolygonNewsClient
from src.data_collector.polygon_data.client import PolygonAPIError


@pytest.mark.unit
def test_extract_article_metadata_maps_fields_correctly():
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    raw = {
        "id": "art-abc",
        "title": "Test Title",
        "description": "Desc",
        "article_url": "http://example.com/a",
        "amp_url": "http://example.com/amp",
        "image_url": "http://img.example.com/x.png",
        "author": "Reporter",
        "published_utc": "2025-08-14T13:45:00Z",
        "keywords": ["k1", "k2"],
        "tickers": ["TICK"],
        "publisher": {
            "name": "Example Pub",
            "homepage_url": "https://pub.example",
            "logo_url": "https://pub.example/logo.png",
            "favicon_url": "https://pub.example/favicon.ico",
        },
        "insights": [{"sentiment": "neutral", "sentiment_reasoning": "Info"}],
    }

    meta = client.extract_article_metadata(raw)
    if meta.get("polygon_id") != "art-abc":
        raise AssertionError("polygon_id not extracted correctly")
    if meta.get("publisher_name") != "Example Pub":
        raise AssertionError("publisher_name not extracted correctly")
    if meta.get("insights", [])[0].get("sentiment") != "neutral":
        raise AssertionError("insight sentiment not extracted correctly")


@pytest.mark.unit
def test_get_news_for_ticker_passes_date_params_and_returns_articles(mocker):
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    captured = {}

    def fake_fetch(self, endpoint, params):
        captured["endpoint"] = endpoint
        captured["params"] = params
        return [{"id": "a1"}, {"id": "a2"}]

    mocker.patch.object(PolygonNewsClient, "_fetch_paginated_data", fake_fetch)

    start = datetime(2025, 8, 1)
    end = datetime(2025, 8, 31)
    res = client.get_news_for_ticker(
        "ACME", published_utc_gte=start, published_utc_lte=end, limit=2
    )

    if not isinstance(res, list):
        raise AssertionError("Expected list of articles from get_news_for_ticker")
    if len(res) != 2:
        raise AssertionError(
            "Unexpected number of articles returned by get_news_for_ticker"
        )
    if "published_utc.gte" not in captured.get("params", {}):
        raise AssertionError("Date parameter not passed to API client")
    if not captured.get("endpoint", "").startswith("/v2/reference/news"):
        raise AssertionError("Unexpected endpoint used for news API")


@pytest.mark.unit
def test_get_news_for_ticker_handles_empty_api_response(mocker):
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    mocker.patch.object(
        PolygonNewsClient, "_fetch_paginated_data", lambda self, e, p: []
    )

    res = client.get_news_for_ticker("ACME")
    if res != []:
        raise AssertionError("Expected empty list when API returns no data")


@pytest.mark.unit
def test_get_news_for_multiple_tickers_handles_failure_and_partial_success(
    mocker, caplog, capsys
):
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    def fake_get(ticker, **kwargs):
        if ticker == "BAD":
            raise PolygonAPIError("simulated")
        return [{"id": f"ok-{ticker}"}]

    mocker.patch.object(
        PolygonNewsClient,
        "get_news_for_ticker",
        lambda self, **kw: fake_get(kw.get("ticker")),
    )

    caplog.clear()
    res = client.get_news_for_multiple_tickers(["GOOD", "BAD"])

    if "GOOD" not in res or not isinstance(res["GOOD"], list):
        raise AssertionError("GOOD results missing or invalid")
    if "BAD" not in res or res["BAD"] != []:
        raise AssertionError("BAD results did not return expected empty list")


@pytest.mark.unit
def test_get_news_by_date_range_filters_and_skips_invalid_dates(mocker, caplog, capsys):
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    articles = [
        {"id": "1", "published_utc": "2025-08-10T00:00:00Z"},
        {"id": "2", "published_utc": "invalid-date"},
        {"id": "3", "published_utc": "2025-09-01T00:00:00Z"},
    ]

    mocker.patch.object(
        PolygonNewsClient, "get_recent_market_news", lambda self, **kw: articles
    )

    start = datetime(2025, 8, 1, tzinfo=timezone.utc)
    end = datetime(2025, 8, 31, tzinfo=timezone.utc)
    caplog.clear()
    res = client.get_news_by_date_range(start, end)

    # should include only id 1
    if not any(a.get("id") == "1" for a in res):
        raise AssertionError("Expected article id '1' in filtered results")
    if any(a.get("id") == "3" for a in res):
        raise AssertionError("Article id '3' should be excluded from filtered results")


@pytest.mark.unit
def test_validate_news_response():
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)
    good = {
        "id": "x",
        "title": "t",
        "article_url": "u",
        "published_utc": "2025-08-14T13:45:00Z",
    }
    bad = {"id": None, "title": "t", "article_url": "u", "published_utc": None}

    if client.validate_news_response(good) is not True:
        raise AssertionError("Good news response validation failed")
    if client.validate_news_response(bad) is not False:
        raise AssertionError("Bad news response validation did not fail as expected")
