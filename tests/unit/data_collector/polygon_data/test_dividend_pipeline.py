from unittest.mock import MagicMock, patch

from src.data_collector.polygon_data.dividend_pipeline import (
    ingest_dividends_for_ticker,
    ingest_dividends_for_all_tickers,
    _ingest_dividends_concurrent,
)


def make_raw(id_suffix: int):
    return {
        "id": f"poly-{id_suffix}",
        "cash_amount": "1.00",
        "ex_dividend_date": "2025-01-01",
        "pay_date": "2025-01-02",
        "declaration_date": "2024-12-01",
        "record_date": "2025-01-03",
        "currency": "USD",
        "frequency": 1,
        "type": "CASH",
    }


@patch("src.database.db_utils._upsert_dividends_batch")
@patch("src.data_collector.polygon_data.dividend_pipeline.transform_dividend_record")
def test_ingest_dividends_for_ticker_success(mock_transform, mock_upsert):
    client = MagicMock()

    # Mock client.get_dividends to return two records
    client.get_dividends.return_value = [make_raw(1), make_raw(2)]

    # ticker dict
    ticker_dict = {"id": 99, "ticker": "AAPL"}

    # Mock transform to return transformed dicts
    mock_transform.side_effect = [
        {"id": "poly-1", "ticker_id": 99, "cash_amount": 1, "raw_payload": {}},
        {"id": "poly-2", "ticker_id": 99, "cash_amount": 1, "raw_payload": {}},
    ]

    # Mock _upsert_dividends_batch to return number of rows
    mock_upsert.return_value = 2

    stats = ingest_dividends_for_ticker(client, ticker_dict, batch_size=2)

    assert stats["fetched"] == 2
    assert stats["upserted"] == 2
    assert stats["invalid"] == 0
    assert stats["skipped"] == 0


@patch("src.data_collector.polygon_data.dividend_pipeline._ingest_dividends_concurrent")
@patch("src.data_collector.polygon_data.dividend_pipeline._ingest_dividends_sequential")
@patch("src.data_collector.polygon_data.dividend_pipeline.DataStorage")
def test_ingest_dividends_for_all_tickers_routing(
    mock_storage_class, mock_sequential, mock_concurrent
):
    """Test that the main function routes correctly between sequential and concurrent."""
    mock_storage = MagicMock()
    mock_storage_class.return_value = mock_storage
    mock_storage.get_tickers.return_value = [{"id": 1, "ticker": "AAPL"}]

    # Test sequential (default)
    ingest_dividends_for_all_tickers(concurrent=False)
    mock_sequential.assert_called_once_with(tickers=[{"id": 1, "ticker": "AAPL"}], batch_size=100)
    mock_concurrent.assert_not_called()

    # Reset mocks
    mock_sequential.reset_mock()
    mock_concurrent.reset_mock()

    # Test concurrent
    ingest_dividends_for_all_tickers(concurrent=True, max_workers=3, requests_per_minute=10)
    mock_concurrent.assert_called_once_with(
        tickers=[{"id": 1, "ticker": "AAPL"}], batch_size=100, max_workers=3, requests_per_minute=10
    )
    mock_sequential.assert_not_called()


@patch("concurrent.futures.ThreadPoolExecutor")
@patch("src.data_collector.polygon_data.dividend_pipeline.PolygonDataClient")
@patch("src.data_collector.polygon_data.dividend_pipeline.DataStorage")
def test_concurrent_processing_basic(mock_storage_class, mock_client_class, mock_executor_class):
    """Test basic concurrent processing functionality."""
    # Mock storage
    mock_storage = MagicMock()
    mock_storage_class.return_value = mock_storage

    # Mock client
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Mock executor
    mock_executor = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = {"fetched": 5, "upserted": 3, "invalid": 0, "skipped": 2}
    mock_executor.__enter__.return_value = mock_executor
    mock_executor.__exit__.return_value = None
    mock_executor.submit.return_value = mock_future
    mock_executor_class.return_value = mock_executor

    # Mock ingest_dividends_for_ticker to return stats
    with patch(
        "src.data_collector.polygon_data.dividend_pipeline.ingest_dividends_for_ticker"
    ) as mock_ingest:
        mock_ingest.return_value = {"fetched": 5, "upserted": 3, "invalid": 0, "skipped": 2}

        tickers = [{"id": 1, "ticker": "AAPL"}, {"id": 2, "ticker": "GOOGL"}]

        result = _ingest_dividends_concurrent(
            tickers=tickers, batch_size=50, max_workers=2, requests_per_minute=5
        )

        # Verify the result contains expected statistics
        assert result["tickers_processed"] == 2
        assert result["total_fetched"] == 10  # 5 + 5
        assert result["total_upserted"] == 6  # 3 + 3
