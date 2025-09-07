import pytest

from src.data_collector.polygon_news.storage import PolygonNewsStorage
from src.data_collector.polygon_news import models
import json


@pytest.mark.unit
def test_store_article_invalid_data_returns_none(
    db_session, sample_raw_article_missing
):
    storage = PolygonNewsStorage(db_session)
    # missing required fields -> validate_article_data should reject
    result = storage.store_article(sample_raw_article_missing)
    assert result is None


@pytest.mark.unit
def test_store_article_create_and_update_flow(db_session, sample_raw_article_full):
    storage = PolygonNewsStorage(db_session)

    # First insert
    article_id = storage.store_article(sample_raw_article_full)
    assert article_id is not None

    # Insert same article should trigger update path and return same id
    article_id2 = storage.store_article(sample_raw_article_full)
    assert article_id2 == article_id


@pytest.mark.unit
def test_store_articles_batch_mixed(
    db_session, sample_raw_article_full, sample_raw_article_missing
):
    storage = PolygonNewsStorage(db_session)
    # Batch: first valid, second invalid -> expect 1 new, 1 failed/skipped
    batch = [sample_raw_article_full, sample_raw_article_missing]
    stats = storage.store_articles_batch(batch)
    assert isinstance(stats, dict)
    assert stats.get("new_articles", 0) >= 1
    # Depending on implementation, invalid article may count as failed_articles or skipped_articles
    assert (stats.get("skipped_articles", 0) + stats.get("failed_articles", 0)) >= 1


@pytest.mark.unit
def test_store_article_keywords_normalization(db_session, sample_raw_article_full):
    """Ensure keywords are normalized to a Python list whether passed as list or JSON string."""
    storage = PolygonNewsStorage(db_session)

    # Case: keywords already a list
    article = sample_raw_article_full.copy()
    article["polygon_id"] = "art-keywords-list"
    article["keywords"] = ["one", "two"]
    aid = storage.store_article(article)
    assert aid is not None

    stored = db_session.query(models.PolygonNewsArticle).filter_by(id=aid).first()
    assert isinstance(stored.keywords, list)
    assert stored.keywords == ["one", "two"]

    # Case: keywords provided as JSON string
    article2 = sample_raw_article_full.copy()
    article2["polygon_id"] = "art-keywords-json"
    article2["keywords"] = json.dumps(["alpha", "beta"])
    aid2 = storage.store_article(article2)
    assert aid2 is not None

    stored2 = db_session.query(models.PolygonNewsArticle).filter_by(id=aid2).first()
    assert isinstance(stored2.keywords, list)
    assert stored2.keywords == ["alpha", "beta"]
