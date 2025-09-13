import pytest
from types import SimpleNamespace
from datetime import date
from unittest.mock import patch

from src.data_collector.polygon_data.data_fetcher import HistoricalDataFetcher
from src.data_collector.polygon_data.data_validator import (
    OHLCVRecord,
    DataQualityMetrics,
)
from src.data_collector.polygon_data.client import PolygonDataClient
from tests._fixtures import canned_api_factory
from tests._fixtures.remote_api_responses import canned_api_factory
from tests._fixtures.factories import polygon_payload_dict


def test_get_historical_data_returns_empty_when_no_data(polygon_client):
    """Return empty records and metrics when API returns no aggregates."""
    empty_resp = canned_api_factory("empty")
    with patch.object(
        PolygonDataClient,
        "get_aggregates",
        return_value=empty_resp.json().get("results", []),
    ):
        fetcher = HistoricalDataFetcher(client=polygon_client)
        records, metrics = fetcher.get_historical_data(
            "TICK", "2020-01-01", "2020-01-02"
        )

    assert records == []
    assert isinstance(metrics, DataQualityMetrics)


def test_get_historical_data_validation_path(polygon_client):
    """Use provided validator to transform raw API payload into validated records."""
    # use polygon_client fixture

    fake_validated = ["vrec1", "vrec2"]
    fake_metrics = SimpleNamespace(
        success_rate=99.9, validation_errors=[], data_gaps=[], outliers=[]
    )
    validator = SimpleNamespace(
        validate_ohlcv_batch=lambda raw, ticker: (fake_validated, fake_metrics)
    )

    resp = canned_api_factory("empty")
    resp._payload = {"results": [{"dummy": True}]}
    with patch.object(
        PolygonDataClient, "get_aggregates", return_value=resp.json().get("results", [])
    ):
        fetcher = HistoricalDataFetcher(client=polygon_client, validator=validator)
        records, metrics = fetcher.get_historical_data(
            "TICK", "2020-01-01", "2020-01-02"
        )

    assert records == fake_validated
    assert metrics.success_rate == 99.9


def test_get_historical_data_transform_without_validation(polygon_client):
    """Transform raw polygon record into OHLCVRecord when validation is disabled."""
    rec = polygon_payload_dict()
    resp = canned_api_factory("empty")
    resp._payload = {"results": [rec]}
    with patch.object(
        PolygonDataClient, "get_aggregates", return_value=resp.json().get("results", [])
    ):
        fetcher = HistoricalDataFetcher(client=polygon_client)
        records, metrics = fetcher.get_historical_data(
            "TICK", "2020-01-01", "2020-01-01", validate_data=False
        )

    assert len(records) == 1
    assert isinstance(records[0], OHLCVRecord)
    assert records[0].ticker == "TICK"
    assert metrics.total_records == 1
    assert metrics.valid_records == 1


def test_get_historical_data_raises_on_client_exception(polygon_client):
    """Raise an exception when the underlying client fails during fetch."""
    with patch.object(
        PolygonDataClient, "get_aggregates", side_effect=Exception("API down")
    ):
        fetcher = HistoricalDataFetcher(client=polygon_client)
        with pytest.raises(Exception):
            fetcher.get_historical_data("TICK", "2020-01-01", "2020-01-02")


def test_get_grouped_daily_data_without_validation(polygon_client):
    """Return grouped OHLCVRecord objects without invoking validator."""
    grouped_rec = polygon_payload_dict(T="AAA", o=1.0, h=2.0, l=0.5, c=1.5, v=200)
    resp = canned_api_factory("empty")
    resp._payload = {"results": [grouped_rec]}
    with patch.object(
        PolygonDataClient,
        "get_grouped_daily",
        return_value=resp.json().get("results", []),
    ):
        fetcher = HistoricalDataFetcher(client=polygon_client)
        results = fetcher.get_grouped_daily_data("2020-01-01", validate_data=False)

    assert "AAA" in results
    assert isinstance(results["AAA"], OHLCVRecord)


def test_get_bulk_historical_data_batches(polygon_client):
    """Fetch multiple tickers in batches and return per-ticker results and metrics."""
    rec = polygon_payload_dict()
    resp = canned_api_factory("empty")
    resp._payload = {"results": [rec]}
    with patch.object(
        PolygonDataClient, "get_aggregates", return_value=resp.json().get("results", [])
    ):
        tickers = ["A", "B", "C"]
        fetcher = HistoricalDataFetcher(client=polygon_client)

        results = fetcher.get_bulk_historical_data(
            tickers, "2020-01-01", "2020-01-02", batch_size=2, delay_between_batches=0
        )

    assert set(results.keys()) == set(tickers)
    for _, (recs, metrics) in results.items():
        assert isinstance(metrics, DataQualityMetrics)


def test_get_historical_data_no_data(polygon_client):
    """Return no records and zero total_records when API returns empty results."""
    empty_resp = canned_api_factory("empty")
    with patch.object(
        PolygonDataClient,
        "get_aggregates",
        return_value=empty_resp.json().get("results", []),
    ):
        fetcher = HistoricalDataFetcher(client=polygon_client)
        records, metrics = fetcher.get_historical_data(
            "AAA", "2020-01-01", "2020-01-02"
        )

    assert records == []
    assert metrics.total_records == 0


def test_get_historical_data_transforms_and_validates(polygon_client):
    """Transform raw polygon records and report consistent quality metrics after validation."""
    # polygon-style records
    raw = [
        polygon_payload_dict(T=None, t=1577923200000, o=10, h=12, l=9, c=11, v=100),
        polygon_payload_dict(T=None, t=1578009600000, o=11, h=13, l=10, c=12, v=200),
    ]
    resp = canned_api_factory("empty")
    resp._payload = {"results": raw}
    with patch.object(
        PolygonDataClient, "get_aggregates", return_value=resp.json().get("results", [])
    ):
        fetcher = HistoricalDataFetcher(client=polygon_client)
        records, metrics = fetcher.get_historical_data(
            "BBB", date(2020, 1, 1), date(2020, 1, 2)
        )

    assert metrics.total_records == 2
    assert metrics.valid_records == len(records)
    assert all(r.ticker == "BBB" for r in records)


def test_get_grouped_daily_data(polygon_client):
    """Fetch grouped daily data and return correctly keyed OHLCVRecord entries."""
    grouped = [
        {"T": "CCC", "t": 1577923200000, "o": 5, "h": 6, "l": 4, "c": 5, "v": 50},
    ]
    resp = canned_api_factory("empty")
    resp._payload = {"results": grouped}
    with patch.object(
        PolygonDataClient,
        "get_grouped_daily",
        return_value=resp.json().get("results", []),
    ):
        fetcher = HistoricalDataFetcher(client=polygon_client)
        res = fetcher.get_grouped_daily_data("2020-01-01")

    assert "CCC" in res
    assert res["CCC"].ticker == "CCC"


@pytest.mark.parametrize("case", ["valid", "invalid"])
def test_transform_polygon_record_payloads(case):
    fetcher = HistoricalDataFetcher(client=None)

    if case == "valid":
        payload = polygon_payload_dict()
        valid = True
    else:
        payload = {"t": None, "o": 0.0, "h": 0.0, "l": 0.0, "c": 0.0, "v": -1}
        valid = False

    transformed = fetcher._transform_polygon_record(payload, ticker="TST")

    if valid:
        assert transformed is not None
        # Basic schema checks
        assert transformed["ticker"] == "TST"
        assert "open" in transformed and isinstance(transformed["open"], float)
    else:
        assert transformed is None
