import pytest
from decimal import Decimal
from datetime import date

from src.data_collector.polygon_data.dividend_pipeline import (
    transform_dividend_record,
    TransformError,
    SkipRecord,
)


def make_valid_raw():
    return {
        "id": "poly-123",
        "cash_amount": "0.50",
        "ex_dividend_date": "2025-01-01",
        "pay_date": "2025-01-02",
        "declaration_date": "2024-12-01",
        "record_date": "2025-01-03",
        "currency": "USD",
        "frequency": 1,
        "type": "CASH",
    }


def test_transform_valid_record():
    raw = make_valid_raw()
    out = transform_dividend_record(raw, ticker_id=42)

    assert out["id"] == "poly-123"
    assert out["ticker_id"] == 42
    assert isinstance(out["cash_amount"], Decimal)
    assert out["cash_amount"] == Decimal("0.50")
    assert out["ex_dividend_date"] == date(2025, 1, 1)
    assert out["raw_payload"]["id"] == raw["id"]


@pytest.mark.parametrize("bad_cash", [None, "not-a-number", ""])
def test_transform_invalid_cash(bad_cash):
    raw = make_valid_raw()
    raw["cash_amount"] = bad_cash
    with pytest.raises(TransformError):
        transform_dividend_record(raw, ticker_id=1)


def test_transform_missing_id_raises_skip():
    raw = make_valid_raw()
    raw.pop("id")
    with pytest.raises(SkipRecord):
        transform_dividend_record(raw, ticker_id=1)


def test_transform_invalid_currency_raises():
    raw = make_valid_raw()
    raw["currency"] = "US"
    with pytest.raises(TransformError):
        transform_dividend_record(raw, ticker_id=1)
