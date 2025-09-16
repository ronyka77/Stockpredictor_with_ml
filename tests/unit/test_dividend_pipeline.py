import pytest
from unittest.mock import MagicMock, patch

from src.data_collector.polygon_data.dividend_pipeline import (
    ingest_dividends_for_ticker,
)


def make_raw(id_suffix: int):
    """
    Build a mock raw dividend record for tests.
    
    Returns a dict shaped like a Polygon dividend payload with predictable fields useful for unit tests.
    Parameters:
        id_suffix (int): Numeric suffix appended to the "id" field (resulting id like "poly-1").
    
    Returns:
        dict: Mock raw dividend record containing string-form amounts and ISO date strings for fields
              such as `ex_dividend_date`, `pay_date`, `declaration_date`, `record_date`, plus
              `currency`, `frequency`, and `type`.
    """
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


@patch("src.data_collector.polygon_data.dividend_pipeline.transform_dividend_record")
def test_ingest_dividends_for_ticker_success(mock_transform):
    client = MagicMock()
    storage = MagicMock()

    # Mock client.get_dividends to return two records
    client.get_dividends.return_value = [make_raw(1), make_raw(2)]

    # storage.get_tickers should return a list with a dict containing id
    storage.get_tickers.return_value = [{"id": 99}]

    # Mock transform to return transformed dicts
    mock_transform.side_effect = [
        {"id": "poly-1", "ticker_id": 99, "cash_amount": 1, "raw_payload": {}},
        {"id": "poly-2", "ticker_id": 99, "cash_amount": 1, "raw_payload": {}},
    ]

    # Mock storage._upsert_dividends_batch to return number of rows
    storage._upsert_dividends_batch.return_value = 2

    stats = ingest_dividends_for_ticker(client, storage, "AAPL", batch_size=2)

    assert stats["fetched"] == 2
    assert stats["upserted"] == 2
    assert stats["invalid"] == 0
    assert stats["skipped"] == 0


