import pytest
from datetime import datetime, timedelta, date

from src.data_collector.polygon_data.data_validator import (
    OHLCVRecord,
    DataValidator,
    TickerInfo,
)


def test_ohlcv_record_valid_and_invalid():
    # valid record should construct and normalize ticker
    valid = {
        "ticker": "aapl",
        "timestamp": "2020-01-02",
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 1000,
        "vwap": 102.0,
    }

    rec = OHLCVRecord(**valid)
    if rec.ticker != "AAPL":
        raise AssertionError("Ticker normalization failed for valid record")
    if not isinstance(rec.timestamp, date):
        raise AssertionError("Timestamp not parsed to date for valid record")

    # invalid: high lower than low should raise
    bad = valid.copy()
    bad.update({"high": 80.0, "low": 90.0})
    with pytest.raises(ValueError):
        OHLCVRecord(**bad)


def test_vwap_fallback_on_invalid_vwap(caplog):
    # VWAP <= 0 should trigger fallback calculation (no exception)
    data = {
        "ticker": "msft",
        "timestamp": "2020-01-02",
        "open": 100.0,
        "high": 120.0,
        "low": 90.0,
        "close": 110.0,
        "volume": 500,
        # Provide a VWAP that is unreasonably far from the day's range to trigger fallback
        "vwap": 1000.0,
    }

    rec = OHLCVRecord(**data)
    # fallback uses (high + low + 2*close) / 4 when volume>0 and then rounds
    expected = round((120.0 + 90.0 + 2 * 110.0) / 4, 4)
    if rec.vwap != expected:
        raise AssertionError("VWAP fallback calculation mismatch")


def test_validate_ohlcv_record_polygon_transform_and_ticker_addition():
    dv = DataValidator(strict_mode=True)

    # Polygon-style payload with milliseconds timestamp and no ticker in payload
    polygon_record = {"t": 1577923200000, "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}
    validated = dv.validate_ohlcv_record(polygon_record, ticker="TSLA")
    if not isinstance(validated, OHLCVRecord):
        raise AssertionError("validate_ohlcv_record did not return OHLCVRecord")
    if validated.ticker != "TSLA":
        raise AssertionError("Ticker not preserved when provided to validator")
    if not isinstance(validated.timestamp, datetime):
        raise AssertionError("Timestamp not parsed to datetime in validator")


def test_validate_ohlcv_batch_metrics_and_outliers_and_gaps():
    dv = DataValidator(strict_mode=False)

    # Build 12 daily records to allow outlier detection (>10 required)
    base_date = datetime(2020, 1, 1)
    records = []
    for i in range(12):
        # introduce a large gap between day 4 and day 9
        dt = base_date + timedelta(days=i + (5 if i >= 9 else 0))
        open_p = 100.0 + i
        close_p = 100.0 + i
        # make one day a big jump to trigger price outlier (>20%)
        if i == 6:
            close_p = 200.0

        records.append(
            {
                "ticker": "ABC",
                "timestamp": dt,
                "open": open_p,
                "high": max(open_p, close_p) + 1,
                "low": min(open_p, close_p) - 1,
                "close": close_p,
                "volume": 1000 + i,
            }
        )

    validated_records, metrics = dv.validate_ohlcv_batch(records, ticker=None)
    if metrics.total_records != len(records):
        raise AssertionError("Metrics total_records mismatch")
    if metrics.valid_records != len(validated_records):
        raise AssertionError("Metrics valid_records mismatch")
    # We expect at least one gap (created by the artificial jump in dates)
    if len(metrics.data_gaps) < 1:
        raise AssertionError("Expected at least one data gap in metrics")
    # We expect price outlier detection to have recorded at least one outlier
    if len(metrics.outliers) < 1:
        raise AssertionError("Expected at least one outlier in metrics")


def test_calculate_batch_vwap():
    # small sample set
    recs = [
        {"high": 10.0, "low": 8.0, "close": 9.0, "volume": 100},
        {"high": 20.0, "low": 18.0, "close": 19.0, "volume": 200},
    ]

    vwap = DataValidator.calculate_batch_vwap(recs)
    # manual calculation: typical prices = 9, 19 => weighted = (9*100 + 19*200) / 300 = (900 + 3800)/300 = 4700/300 = 15.6667
    if round(vwap, 4) != round(4700 / 300, 4):
        raise AssertionError("Batch VWAP calculation mismatch")


def test_validate_ticker_empty_raises():
    with pytest.raises(ValueError):
        OHLCVRecord(
            ticker="",
            timestamp="2020-01-01",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1,
        )


def test_validate_ticker_clean_uppercase():
    rec = OHLCVRecord(
        ticker=" aapl ",
        timestamp="2020-01-01",
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1,
    )
    if rec.ticker != "AAPL":
        raise AssertionError("Ticker cleanup/uppercase failed")


def test_high_lower_consistency_raises():
    # high < low should raise
    with pytest.raises(ValueError):
        OHLCVRecord(
            ticker="A",
            timestamp="2020-01-01",
            open=10.0,
            high=5.0,
            low=6.0,
            close=7.0,
            volume=1,
        )


def test_low_greater_than_open_or_close_raises():
    # low > open should raise
    with pytest.raises(ValueError):
        OHLCVRecord(
            ticker="A",
            timestamp="2020-01-01",
            open=10.0,
            high=20.0,
            low=15.0,
            close=12.0,
            volume=1,
        )


def test_volume_negative_and_too_large_raises():
    with pytest.raises(ValueError):
        OHLCVRecord(
            ticker="A",
            timestamp="2020-01-01",
            open=1.0,
            high=2.0,
            low=1.0,
            close=1.5,
            volume=-1,
        )

    with pytest.raises(ValueError):
        OHLCVRecord(
            ticker="A",
            timestamp="2020-01-01",
            open=1.0,
            high=2.0,
            low=1.0,
            close=1.5,
            volume=10**13,
        )


def test_vwap_fallback_on_invalid_value():
    # vwap <= 0 should trigger fallback calculation
    rec = OHLCVRecord(
        ticker="A",
        timestamp="2020-01-01",
        open=10.0,
        high=12.0,
        low=9.0,
        close=11.0,
        volume=100,
        vwap=0,
    )
    expected = OHLCVRecord._calculate_fallback_vwap(12.0, 9.0, 11.0, 100)
    if not isinstance(rec.vwap, float):
        raise AssertionError("VWAP not set to float")
    if rec.vwap != expected:
        raise AssertionError("VWAP fallback calculation mismatch")


def test_adjusted_close_none_ok_and_negative_raises():
    # None is allowed
    rec = OHLCVRecord(
        ticker="A",
        timestamp="2020-01-01",
        open=1.0,
        high=2.0,
        low=1.0,
        close=1.5,
        volume=1,
        adjusted_close=None,
    )
    if rec.adjusted_close is not None:
        raise AssertionError("Adjusted close None expected")

    with pytest.raises(ValueError):
        OHLCVRecord(
            ticker="A",
            timestamp="2020-01-01",
            open=1.0,
            high=2.0,
            low=1.0,
            close=1.5,
            volume=1,
            adjusted_close=-1,
        )


def test_timestamp_parsing_and_future_date_rejected():
    # ISO format with T -> datetime
    rec = OHLCVRecord(
        ticker="A",
        timestamp="2020-01-01T00:00:00Z",
        open=1.0,
        high=2.0,
        low=1.0,
        close=1.5,
        volume=1,
    )
    if not isinstance(rec.timestamp, (datetime, date)):
        raise AssertionError("Timestamp parsing did not produce datetime or date")

    # future date should raise
    future_iso = (datetime.now() + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        OHLCVRecord(
            ticker="A",
            timestamp=future_iso,
            open=1.0,
            high=2.0,
            low=1.0,
            close=1.5,
            volume=1,
        )


def test_tickerinfo_name_empty_raises():
    with pytest.raises(ValueError):
        TickerInfo(ticker="A", name="   ", market="stocks")
