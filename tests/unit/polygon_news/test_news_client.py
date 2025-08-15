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
    assert meta["polygon_id"] == "art-abc"
    assert meta["publisher_name"] == "Example Pub"
    assert meta["insights"][0]["sentiment"] == "neutral"


@pytest.mark.unit
def test_get_news_for_ticker_passes_date_params_and_returns_articles(monkeypatch):
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    captured = {}

    def fake_fetch(self, endpoint, params):
        captured['endpoint'] = endpoint
        captured['params'] = params
        return [{"id": "a1"}, {"id": "a2"}]

    monkeypatch.setattr(PolygonNewsClient, "_fetch_paginated_data", fake_fetch)

    start = datetime(2025, 8, 1)
    end = datetime(2025, 8, 31)
    res = client.get_news_for_ticker("ACME", published_utc_gte=start, published_utc_lte=end, limit=2)

    assert isinstance(res, list)
    assert len(res) == 2
    assert "published_utc.gte" in captured['params']
    assert captured['endpoint'].startswith("/v2/reference/news")


@pytest.mark.unit
def test_get_news_for_ticker_handles_empty_api_response(monkeypatch):
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    monkeypatch.setattr(PolygonNewsClient, "_fetch_paginated_data", lambda self, e, p: [])

    res = client.get_news_for_ticker("ACME")
    assert res == []


@pytest.mark.unit
def test_get_news_for_multiple_tickers_handles_failure_and_partial_success(monkeypatch, caplog):
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    def fake_get(ticker, **kwargs):
        if ticker == "BAD":
            raise PolygonAPIError("simulated")
        return [{"id": f"ok-{ticker}"}]

    monkeypatch.setattr(PolygonNewsClient, "get_news_for_ticker", lambda self, **kw: fake_get(kw.get('ticker')))

    caplog.clear()
    res = client.get_news_for_multiple_tickers(["GOOD", "BAD"])

    assert "GOOD" in res and isinstance(res["GOOD"], list)
    assert "BAD" in res and res["BAD"] == []
    assert any("Failed to fetch news for ticker" in r.message for r in caplog.records if r.levelname in ("WARNING", "ERROR"))


@pytest.mark.unit
def test_get_news_by_date_range_filters_and_skips_invalid_dates(monkeypatch, caplog):
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)

    articles = [
        {"id": "1", "published_utc": "2025-08-10T00:00:00Z"},
        {"id": "2", "published_utc": "invalid-date"},
        {"id": "3", "published_utc": "2025-09-01T00:00:00Z"},
    ]

    monkeypatch.setattr(PolygonNewsClient, "get_recent_market_news", lambda self, **kw: articles)

    start = datetime(2025, 8, 1, tzinfo=timezone.utc)
    end = datetime(2025, 8, 31, tzinfo=timezone.utc)
    caplog.clear()
    res = client.get_news_by_date_range(start, end)

    # should include only id 1
    assert any(a.get('id') == '1' for a in res)
    assert all(a.get('id') != '3' for a in res)
    assert any('Invalid date format' in (getattr(r, 'message', '') or r.getMessage()) for r in caplog.records)


@pytest.mark.unit
def test_validate_news_response():
    client = PolygonNewsClient(api_key="test", requests_per_minute=10)
    good = {"id": "x", "title": "t", "article_url": "u", "published_utc": "2025-08-14T13:45:00Z"}
    bad = {"id": None, "title": "t", "article_url": "u", "published_utc": None}

    assert client.validate_news_response(good) is True
    assert client.validate_news_response(bad) is False


