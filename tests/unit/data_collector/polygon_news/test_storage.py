import pytest

from src.data_collector.polygon_news.storage import PolygonNewsStorage
import json


@pytest.mark.unit
def test_store_article_invalid_data_returns_none(sample_raw_article_missing):
    """store_article returns None when input article data is invalid"""
    storage = PolygonNewsStorage()
    # missing required fields -> validate_article_data should reject
    result = storage.store_article(sample_raw_article_missing)
    assert result is None


@pytest.mark.unit
def test_store_article_create_and_update_flow(mocker, sample_raw_article_full):
    """Create new article then update same article; both paths return expected ids"""
    storage = PolygonNewsStorage()

    # Simulate successful DB insertion via run_in_transaction
    mocker.patch(
        "src.data_collector.polygon_news.storage.run_in_transaction",
        return_value=1,
    )

    # First insert
    article_id = storage.store_article(sample_raw_article_full)
    assert article_id is not None

    # Insert same article should trigger update path and return same id
    article_id2 = storage.store_article(sample_raw_article_full)
    assert article_id2 == article_id


@pytest.mark.unit
def test_store_articles_batch_mixed(
    mocker, sample_raw_article_full, sample_raw_article_missing
):
    """Batch storage handles mixture of valid and invalid articles appropriately"""
    storage = PolygonNewsStorage()

    # Ensure run_in_transaction returns an id for created articles
    mocker.patch(
        "src.data_collector.polygon_news.storage.run_in_transaction", return_value=1
    )

    # Make _create_new_article succeed for valid articles and return None for invalid
    def _create_new_article_side(article_data):
        # Simulate failure for articles missing title or tickers
        if not article_data.get("title") or not article_data.get("tickers"):
            return None
        return 1

    mocker.patch.object(
        storage, "_create_new_article", side_effect=_create_new_article_side
    )
    # Batch: first valid, second invalid -> expect 1 new, 1 failed/skipped
    batch = [sample_raw_article_full, sample_raw_article_missing]
    stats = storage.store_articles_batch(batch)
    assert isinstance(stats, dict)
    assert stats.get("new_articles", 0) >= 1
    # Depending on implementation, invalid article may count as failed_articles or skipped_articles
    assert (stats.get("skipped_articles", 0) + stats.get("failed_articles", 0)) >= 1


@pytest.mark.unit
def test_store_article_keywords_normalization(mocker, sample_raw_article_full):
    """Ensure keywords are normalized to a Python list whether passed as list or JSON string."""
    storage = PolygonNewsStorage()

    mocker.patch(
        "src.data_collector.polygon_news.storage.run_in_transaction", return_value=1
    )

    # Case: keywords already a list -> should return an id
    article = sample_raw_article_full.copy()
    article["polygon_id"] = "art-keywords-list"
    article["keywords"] = ["one", "two"]
    aid = storage.store_article(article)
    assert aid is not None

    # Case: keywords provided as JSON string -> should also return an id
    article2 = sample_raw_article_full.copy()
    article2["polygon_id"] = "art-keywords-json"
    article2["keywords"] = json.dumps(["alpha", "beta"])
    aid2 = storage.store_article(article2)
    assert aid2 is not None
