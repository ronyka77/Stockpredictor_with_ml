import pytest
from datetime import datetime, date, timedelta

from src.data_collector.polygon_data.data_validator import (
    OHLCVRecord,
    TickerInfo,
)


def test_validate_ticker_empty_raises():
    with pytest.raises(ValueError):
        OHLCVRecord(ticker='', timestamp='2020-01-01', open=1.0, high=1.0, low=1.0, close=1.0, volume=1)


def test_validate_ticker_clean_uppercase():
    rec = OHLCVRecord(ticker=' aapl ', timestamp='2020-01-01', open=1.0, high=1.0, low=1.0, close=1.0, volume=1)
    assert rec.ticker == 'AAPL'


def test_high_lower_consistency_raises():
    # high < low should raise
    with pytest.raises(ValueError):
        OHLCVRecord(ticker='A', timestamp='2020-01-01', open=10.0, high=5.0, low=6.0, close=7.0, volume=1)


def test_low_greater_than_open_or_close_raises():
    # low > open should raise
    with pytest.raises(ValueError):
        OHLCVRecord(ticker='A', timestamp='2020-01-01', open=10.0, high=20.0, low=15.0, close=12.0, volume=1)


def test_volume_negative_and_too_large_raises():
    with pytest.raises(ValueError):
        OHLCVRecord(ticker='A', timestamp='2020-01-01', open=1.0, high=2.0, low=1.0, close=1.5, volume=-1)

    with pytest.raises(ValueError):
        OHLCVRecord(ticker='A', timestamp='2020-01-01', open=1.0, high=2.0, low=1.0, close=1.5, volume=10**13)


def test_vwap_fallback_on_invalid_value():
    # vwap <= 0 should trigger fallback calculation
    rec = OHLCVRecord(
        ticker='A',
        timestamp='2020-01-01',
        open=10.0,
        high=12.0,
        low=9.0,
        close=11.0,
        volume=100,
        vwap=0,
    )
    expected = OHLCVRecord._calculate_fallback_vwap(12.0, 9.0, 11.0, 100)
    assert isinstance(rec.vwap, float)
    assert rec.vwap == expected


def test_adjusted_close_none_ok_and_negative_raises():
    # None is allowed
    rec = OHLCVRecord(ticker='A', timestamp='2020-01-01', open=1.0, high=2.0, low=1.0, close=1.5, volume=1, adjusted_close=None)
    assert rec.adjusted_close is None

    with pytest.raises(ValueError):
        OHLCVRecord(ticker='A', timestamp='2020-01-01', open=1.0, high=2.0, low=1.0, close=1.5, volume=1, adjusted_close=-1)


def test_timestamp_parsing_and_future_date_rejected():
    # ISO format with T -> datetime
    rec = OHLCVRecord(ticker='A', timestamp='2020-01-01T00:00:00Z', open=1.0, high=2.0, low=1.0, close=1.5, volume=1)
    assert isinstance(rec.timestamp, (datetime, date))

    # future date should raise
    future_iso = (datetime.now() + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        OHLCVRecord(ticker='A', timestamp=future_iso, open=1.0, high=2.0, low=1.0, close=1.5, volume=1)


def test_tickerinfo_name_empty_raises():
    with pytest.raises(ValueError):
        TickerInfo(ticker='A', name='   ', market='stocks')


