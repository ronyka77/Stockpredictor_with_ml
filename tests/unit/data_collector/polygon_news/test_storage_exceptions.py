import pytest

from src.data_collector.polygon_news.storage import PolygonNewsStorage


@pytest.mark.unit
def test_store_article_on_exception_rolls_back_and_returns_none(mocker, sample_raw_article_full):
    """Adapted to pool-based storage: patch run_in_transaction to raise and
    verify storage returns None without relying on ORM session rollback.
    """
    storage = PolygonNewsStorage()

    # Force transaction runner to raise
    mocker.patch(
        "src.data_collector.polygon_news.storage.run_in_transaction",
        side_effect=RuntimeError("db error"),
    )

    res = storage.store_article(sample_raw_article_full)
    assert res is None
