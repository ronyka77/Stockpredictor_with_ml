from src.data_collector.polygon_fundamentals.client import (
    PolygonFundamentalsClient,
)
from src.data_collector.polygon_fundamentals.data_models import FinancialValue


def test_extract_financial_value_nested_dict():
    """Extract nested dict financial value into FinancialValue model"""
    data = {
        "financials": {
            "income_statement": {"revenues": {"value": 123.45, "unit": "USD"}}
        }
    }

    fv = PolygonFundamentalsClient._extract_financial_value(
        data, "revenues", "income_statement"
    )

    assert isinstance(fv, FinancialValue)
    assert fv.value == 123.45


def test_extract_financial_value_legacy_numeric():
    """Extract legacy numeric financial value into FinancialValue model"""
    data = {"revenues": 200}
    fv = PolygonFundamentalsClient._extract_financial_value(
        data, "revenues", "income_statement"
    )
    assert isinstance(fv, FinancialValue)
    assert fv.value == 200.0


def test_extract_financial_value_bad_cast_returns_none():
    """Non-numeric legacy financial value returns None instead of raising"""
    data = {"revenues": "not-a-number"}
    fv = PolygonFundamentalsClient._extract_financial_value(
        data, "revenues", "income_statement"
    )
    assert fv is None
