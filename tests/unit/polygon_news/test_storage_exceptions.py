import pytest

from src.data_collector.polygon_news.storage import PolygonNewsStorage


@pytest.mark.unit
def test_store_article_on_exception_rolls_back_and_returns_none(db_session, sample_raw_article_full):
    storage = PolygonNewsStorage(db_session)

    # Force _create_new_article to raise
    def raise_create(article_data):
        raise RuntimeError("db error")

    storage._create_new_article = raise_create

    # Patch session.rollback to mark a flag
    def fake_rollback():
        setattr(db_session, "_rolled_back", True)

    db_session.rollback = fake_rollback

    res = storage.store_article(sample_raw_article_full)
    assert res is None
    assert getattr(db_session, "_rolled_back", False) is True


