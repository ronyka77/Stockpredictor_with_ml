import pytest
from datetime import date

from src.data_collector.polygon_data.data_validator import OHLCVRecord
from tests.fixtures.factories import OHLCVRecordFactory


def test_ohlcv_record_valid_and_invalid():
    """Validate OHLCVRecord accepts valid input and rejects invalid ranges"""
    rec = OHLCVRecordFactory.build(
        ticker="aapl",
        timestamp="2020-01-02",
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        volume=1000,
        vwap=102.0,
    )

    assert rec.ticker == "AAPL"
    assert isinstance(rec.timestamp, date)

    bad = rec.model_dump() if hasattr(rec, "model_dump") else rec.dict()
    bad.update({"high": 80.0, "low": 90.0})
    with pytest.raises(ValueError):
        OHLCVRecord(**bad)


def test_vwap_fallback_on_invalid_vwap(caplog):
    """Ensure invalid VWAP is replaced by computed fallback value"""
    data = {
        "ticker": "msft",
        "timestamp": "2020-01-02",
        "open": 100.0,
        "high": 120.0,
        "low": 90.0,
        "close": 110.0,
        "volume": 500,
        "vwap": 1000.0,
    }

    rec = OHLCVRecord(**data)
    expected = round((120.0 + 90.0 + 2 * 110.0) / 4, 4)
    assert rec.vwap == expected
