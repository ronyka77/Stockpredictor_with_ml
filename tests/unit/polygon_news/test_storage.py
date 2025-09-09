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
    if result is not None:
        raise AssertionError("Expected None result for invalid article data")


@pytest.mark.unit
def test_store_article_create_and_update_flow(db_session, sample_raw_article_full):
    storage = PolygonNewsStorage(db_session)

    # First insert
    article_id = storage.store_article(sample_raw_article_full)
    if article_id is None:
        raise AssertionError("Expected article_id to be returned on successful store")

    # Insert same article should trigger update path and return same id
    article_id2 = storage.store_article(sample_raw_article_full)
    if article_id2 != article_id:
        raise AssertionError(
            "Second store did not return same article_id on update path"
        )


@pytest.mark.unit
def test_store_articles_batch_mixed(
    db_session, sample_raw_article_full, sample_raw_article_missing
):
    storage = PolygonNewsStorage(db_session)
    # Batch: first valid, second invalid -> expect 1 new, 1 failed/skipped
    batch = [sample_raw_article_full, sample_raw_article_missing]
    stats = storage.store_articles_batch(batch)
    if not isinstance(stats, dict):
        raise AssertionError("Expected stats to be a dict")
    if stats.get("new_articles", 0) < 1:
        raise AssertionError("Expected at least one new article in batch stats")
    # Depending on implementation, invalid article may count as failed_articles or skipped_articles
    if (stats.get("skipped_articles", 0) + stats.get("failed_articles", 0)) < 1:
        raise AssertionError(
            "Expected at least one skipped or failed article in batch stats"
        )


@pytest.mark.unit
def test_store_article_keywords_normalization(db_session, sample_raw_article_full):
    """Ensure keywords are normalized to a Python list whether passed as list or JSON string."""
    storage = PolygonNewsStorage(db_session)

    # Case: keywords already a list
    article = sample_raw_article_full.copy()
    article["polygon_id"] = "art-keywords-list"
    article["keywords"] = ["one", "two"]
    aid = storage.store_article(article)
    if aid is None:
        raise AssertionError(
            "Expected aid to be returned when storing article with keywords list"
        )

    stored = db_session.query(models.PolygonNewsArticle).filter_by(id=aid).first()
    if not isinstance(stored.keywords, list):
        raise AssertionError("Stored keywords should be a list")
    if stored.keywords != ["one", "two"]:
        raise AssertionError("Stored keywords did not match expected list")

    # Case: keywords provided as JSON string
    article2 = sample_raw_article_full.copy()
    article2["polygon_id"] = "art-keywords-json"
    article2["keywords"] = json.dumps(["alpha", "beta"])
    aid2 = storage.store_article(article2)
    if aid2 is None:
        raise AssertionError(
            "Expected aid to be returned when storing article with JSON keywords"
        )

    stored2 = db_session.query(models.PolygonNewsArticle).filter_by(id=aid2).first()
    if not isinstance(stored2.keywords, list):
        raise AssertionError("Stored2 keywords should be a list")
    if stored2.keywords != ["alpha", "beta"]:
        raise AssertionError("Stored2 keywords did not match expected list")
