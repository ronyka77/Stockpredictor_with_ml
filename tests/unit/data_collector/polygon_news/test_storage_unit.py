from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.data_collector.polygon_news.storage import PolygonNewsStorage
from tests.unit.data_collector.polygon_news.helpers import (
    processed_article_expected,
)


@pytest.mark.unit
def test__create_new_article_returns_id_on_success():
    """_create_new_article returns DB id when transaction succeeds"""
    storage = PolygonNewsStorage()
    article = processed_article_expected()

    with patch(
        "src.data_collector.polygon_news.storage.run_in_transaction", return_value=42
    ) as mock_tx:
        aid = storage._create_new_article(article)

    assert aid == 42
    mock_tx.assert_called_once()


@pytest.mark.unit
def test__create_new_article_invalid_published_date_returns_none():
    """Return None when published_utc cannot be parsed; avoid DB calls"""
    storage = PolygonNewsStorage()
    article = processed_article_expected()
    article["published_utc"] = "not-a-date"

    # No DB interaction expected; ensure None is returned
    with patch("src.data_collector.polygon_news.storage.run_in_transaction") as mock_tx:
        res = storage._create_new_article(article)

    assert res is None
    assert mock_tx.call_count == 0


@pytest.mark.unit
def test__update_existing_article_updates_and_returns_true():
    """Update path applies DB changes and returns True when changes occur"""
    storage = PolygonNewsStorage()

    existing = {
        "id": 1,
        "title": "old",
        "description": "old",
        "quality_score": None,
        "relevance_score": None,
        "keywords": [],
    }

    new = processed_article_expected()
    # change several fields to force updates
    new["title"] = "new title"
    new["description"] = "new desc"
    new["keywords"] = ["k"]
    new["tickers"] = ["A", "B"]
    new["insights"] = [{"sentiment": "positive", "sentiment_reasoning": "ok"}]

    with patch("src.data_collector.polygon_news.storage.execute") as mock_exec:
        updated = storage._update_existing_article(existing, new)

    assert updated is True
    # execute should be called for update + delete/insert of tickers + insert insights
    assert mock_exec.call_count >= 3


@pytest.mark.unit
def test__parse_datetime_variants_and_failure():
    """Parse various datetime inputs and return None on failure"""
    storage = PolygonNewsStorage()

    # datetime object passthrough
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert storage._parse_datetime(dt) == dt

    # ISO Z string
    s = "2025-08-14T13:45:00Z"
    parsed = storage._parse_datetime(s)
    assert parsed is not None

    # bad format
    assert storage._parse_datetime("oops") is None


@pytest.mark.unit
def test_get_latest_date_for_ticker_and_overall():
    storage = PolygonNewsStorage()
    dt = datetime(2025, 8, 1, tzinfo=timezone.utc)

    with patch(
        "src.data_collector.polygon_news.storage.fetch_one", return_value={"latest": dt}
    ):
        res = storage.get_latest_date_for_ticker("AAPL")
        assert res == dt

    with patch(
        "src.data_collector.polygon_news.storage.fetch_one", return_value={"latest": dt}
    ):
        res2 = storage.get_latest_date_overall()
        assert res2 == dt


@pytest.mark.unit
def test_cleanup_old_articles_calls_execute_and_returns_zero():
    storage = PolygonNewsStorage()

    with patch("src.data_collector.polygon_news.storage.execute") as mock_exec:
        res = storage.cleanup_old_articles(retention_days=1)

    assert res == 0
    mock_exec.assert_called_once()


@pytest.mark.unit
def test_get_articles_for_ticker_calls_fetch_all_and_returns_rows():
    storage = PolygonNewsStorage()
    rows = [{"id": 1}]

    with patch(
        "src.data_collector.polygon_news.storage.fetch_all",
        return_value=rows,
    ) as mock_fetch:
        out = storage.get_articles_for_ticker("ACME", limit=5)

    assert out == rows
    mock_fetch.assert_called_once()


@pytest.mark.unit
def test_get_article_statistics_success_path():
    # Prepare controlled returns for fetch_one and fetch_all
    storage = PolygonNewsStorage()
    total = {"cnt": 5}
    tickers = [{"ticker": "A", "cnt": 2}]
    sentiments = [{"sentiment": "positive", "cnt": 3}]
    publishers = [{"publisher_name": "P", "cnt": 2}]

    def fake_fetch_one(sql, params=None, dict_cursor=False):
        # total query
        if "COUNT(*) as cnt" in sql:
            return total
        return None

    def fake_fetch_all(sql, params=None, dict_cursor=False):
        if "FROM polygon_news_tickers" in sql:
            return tickers
        if "FROM polygon_news_insights" in sql:
            return sentiments
        if "FROM polygon_news_articles GROUP BY publisher_name" in sql:
            return publishers
        return []

    with patch("src.data_collector.polygon_news.storage.fetch_one", fake_fetch_one), patch(
        "src.data_collector.polygon_news.storage.fetch_all", fake_fetch_all
    ), patch.object(
        storage, "get_latest_date_overall", lambda: datetime(2025, 8, 1, tzinfo=timezone.utc)
    ):

        stats = storage.get_article_statistics()

    assert stats["total_articles"] == 5
    assert "top_tickers" in stats
    assert "sentiment_distribution" in stats
    assert "top_publishers" in stats


@pytest.mark.unit
def test_health_check_unhealthy_on_exception():
    storage = PolygonNewsStorage()

    def raise_exc(*a, **k):
        raise RuntimeError("db")

    with patch("src.data_collector.polygon_news.storage.fetch_one", raise_exc):
        res = storage.health_check()
        assert res.get("status") == "unhealthy"
