import pytest
from types import SimpleNamespace
from datetime import datetime, date

from src.data_collector.polygon_data.data_fetcher import HistoricalDataFetcher
from src.data_collector.polygon_data.data_validator import (
    OHLCVRecord,
    DataQualityMetrics,
)
from tests._fixtures.conftest import mock_http_client

def _make_polygon_record(day_index=0):
    # Create a simple valid polygon-style record with millisecond timestamp
    ts = int(datetime(2020, 1, 1 + day_index).timestamp() * 1000)
    return {
        "t": ts,
        "o": 10.0 + day_index,
        "h": 11.0 + day_index,
        "l": 9.0 + day_index,
        "c": 10.5 + day_index,
        "v": 1000 + day_index,
    }


def test_get_historical_data_returns_empty_when_no_data(mock_http_client):
    mock_http_client.set_aggregates([])

    fetcher = HistoricalDataFetcher(client=mock_http_client)
    records, metrics = fetcher.get_historical_data("TICK", "2020-01-01", "2020-01-02")

    if records != []:
        raise AssertionError("Expected no records when API returns empty aggregates")
    if not isinstance(metrics, DataQualityMetrics):
        raise AssertionError("Metrics should be DataQualityMetrics")


def test_get_historical_data_validation_path(mock_http_client):
    # Client returns raw payload but validator will be invoked and its output returned
    mock_http_client.set_aggregates([{"dummy": True}])

    fake_validated = ["vrec1", "vrec2"]
    fake_metrics = SimpleNamespace(
        success_rate=99.9, validation_errors=[], data_gaps=[], outliers=[]
    )

    validator = SimpleNamespace(
        validate_ohlcv_batch=lambda raw, ticker: (fake_validated, fake_metrics)
    )

    fetcher = HistoricalDataFetcher(client=mock_http_client, validator=validator)
    records, metrics = fetcher.get_historical_data("TICK", "2020-01-01", "2020-01-02")

    if records != fake_validated:
        raise AssertionError("Validator path did not return validated records")
    if metrics.success_rate != 99.9:
        raise AssertionError("Metrics success_rate mismatch in validator path")


def test_get_historical_data_transform_without_validation(mock_http_client):
    # Client returns one valid polygon record and validate_data=False should transform into OHLCVRecord
    rec = _make_polygon_record()
    mock_http_client.set_aggregates([rec])

    fetcher = HistoricalDataFetcher(client=mock_http_client)
    records, metrics = fetcher.get_historical_data(
        "TICK", "2020-01-01", "2020-01-01", validate_data=False
    )

    if len(records) != 1:
        raise AssertionError("Expected one record in transformed results")
    if not isinstance(records[0], OHLCVRecord):
        raise AssertionError("Transformed record is not OHLCVRecord")
    if records[0].ticker != "TICK":
        raise AssertionError("Transformed record ticker mismatch")
    if metrics.total_records != 1:
        raise AssertionError("Metrics total_records mismatch")
    if metrics.valid_records != 1:
        raise AssertionError("Metrics valid_records mismatch")


def test_get_historical_data_raises_on_client_exception(mock_http_client):
    mock_http_client.set_aggregates(Exception("API down"))

    fetcher = HistoricalDataFetcher(client=mock_http_client)
    with pytest.raises(Exception):
        fetcher.get_historical_data("TICK", "2020-01-01", "2020-01-02")


def test_get_grouped_daily_data_without_validation(mock_http_client):
    # Grouped data contains 'T' ticker; validate_data=False path should return OHLCVRecord objects
    grouped_rec = {"T": "AAA", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 200}
    mock_http_client.set_grouped([grouped_rec])

    fetcher = HistoricalDataFetcher(client=mock_http_client)
    results = fetcher.get_grouped_daily_data("2020-01-01", validate_data=False)

    if "AAA" not in results:
        raise AssertionError("Expected grouped results to contain 'AAA'")
    if not isinstance(results["AAA"], OHLCVRecord):
        raise AssertionError("Grouped result value is not OHLCVRecord")


def test_get_bulk_historical_data_batches(mock_http_client):
    # Ensure batching works and returns results for each ticker
    rec = _make_polygon_record()
    mock_http_client.set_aggregates([rec])

    tickers = ["A", "B", "C"]
    fetcher = HistoricalDataFetcher(client=mock_http_client)

    results = fetcher.get_bulk_historical_data(
        tickers, "2020-01-01", "2020-01-02", batch_size=2, delay_between_batches=0
    )

    if set(results.keys()) != set(tickers):
        raise AssertionError(
            "Bulk historical data did not return results for all requested tickers"
        )
    for k, (recs, metrics) in results.items():
        # Each returned list may contain OHLCVRecord objects after validation, or be empty depending on validator
        if not isinstance(metrics, DataQualityMetrics):
            raise AssertionError("Metrics in bulk results should be DataQualityMetrics")


def test_get_historical_data_no_data(mock_http_client):
    mock_http_client.set_aggregates([])
    fetcher = HistoricalDataFetcher(client=mock_http_client)

    records, metrics = fetcher.get_historical_data("AAA", "2020-01-01", "2020-01-02")
    if records != []:
        raise AssertionError("Expected no records when no aggregates returned")
    if metrics.total_records != 0:
        raise AssertionError("Expected metrics.total_records == 0 when no records")


def test_get_historical_data_transforms_and_validates(mock_http_client):
    # polygon-style records
    raw = [
        {"t": 1577923200000, "o": 10, "h": 12, "l": 9, "c": 11, "v": 100},
        {"t": 1578009600000, "o": 11, "h": 13, "l": 10, "c": 12, "v": 200},
    ]
    mock_http_client.set_aggregates(raw)
    fetcher = HistoricalDataFetcher(client=mock_http_client)

    records, metrics = fetcher.get_historical_data(
        "BBB", date(2020, 1, 1), date(2020, 1, 2)
    )
    if metrics.total_records != 2:
        raise AssertionError(
            "Metrics total_records mismatch for transformed/validated path"
        )
    if metrics.valid_records != len(records):
        raise AssertionError(
            "Metrics valid_records mismatch for transformed/validated path"
        )
    if not all(r.ticker == "BBB" for r in records):
        raise AssertionError("Not all transformed records have expected ticker 'BBB'")


def test_get_grouped_daily_data(mock_http_client):
    grouped = [
        {"T": "CCC", "t": 1577923200000, "o": 5, "h": 6, "l": 4, "c": 5, "v": 50},
    ]
    mock_http_client.set_grouped(grouped)
    fetcher = HistoricalDataFetcher(client=mock_http_client)

    res = fetcher.get_grouped_daily_data("2020-01-01")
    if "CCC" not in res:
        raise AssertionError("Expected 'CCC' key in grouped daily results")
    if res["CCC"].ticker != "CCC":
        raise AssertionError("Grouped result ticker mismatch for 'CCC'")
