from unittest.mock import patch


def test_news_client_fetches_and_parses():
    """Verify news client fetches paginated data and parses article dicts"""
    from src.data_collector.polygon_news.news_client import PolygonNewsClient

    client = PolygonNewsClient(api_key="TEST", requests_per_minute=100)

    # Patch _fetch_paginated_data to return a small list
    articles = [
        {"id": "1", "title": "t1", "article_url": "u", "published_utc": "2025-01-01T00:00:00Z"}
    ]
    with patch.object(PolygonNewsClient, "_fetch_paginated_data", return_value=articles):
        res = client.get_news_for_ticker("AAPL")

    assert isinstance(res, list)
    assert res[0]["id"] == "1"
