import pytest
from datetime import date

from src.data_collector.polygon_data.data_fetcher import HistoricalDataFetcher


def test_get_historical_data_no_data(mock_http_client):
    mock_http_client.set_aggregates([])
    fetcher = HistoricalDataFetcher(client=mock_http_client)

    records, metrics = fetcher.get_historical_data("AAA", "2020-01-01", "2020-01-02")
    assert records == []
    assert metrics.total_records == 0


def test_get_historical_data_transforms_and_validates(mock_http_client):
    # polygon-style records
    raw = [
        {"t": 1577923200000, "o": 10, "h": 12, "l": 9, "c": 11, "v": 100},
        {"t": 1578009600000, "o": 11, "h": 13, "l": 10, "c": 12, "v": 200},
    ]
    mock_http_client.set_aggregates(raw)
    fetcher = HistoricalDataFetcher(client=mock_http_client)

    records, metrics = fetcher.get_historical_data("BBB", date(2020, 1, 1), date(2020, 1, 2))
    assert metrics.total_records == 2
    assert metrics.valid_records == len(records)
    assert all(r.ticker == "BBB" for r in records)


def test_get_grouped_daily_data(mock_http_client):
    grouped = [
        {"T": "CCC", "t": 1577923200000, "o": 5, "h": 6, "l": 4, "c": 5, "v": 50},
    ]
    mock_http_client.set_grouped(grouped)
    fetcher = HistoricalDataFetcher(client=mock_http_client)

    res = fetcher.get_grouped_daily_data("2020-01-01")
    assert "CCC" in res
    assert res["CCC"].ticker == "CCC"


