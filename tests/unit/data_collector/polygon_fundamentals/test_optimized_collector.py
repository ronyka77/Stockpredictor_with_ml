import pytest
from dataclasses import dataclass
from datetime import date

from polyfactory.factories import DataclassFactory
from unittest.mock import Mock, patch
import asyncio

from src.data_collector.polygon_fundamentals.optimized_collector import (
    OptimizedFundamentalCollector,
)


@dataclass
class SimpleValue:
    value: float
    source: str | None = None


class SimpleValueFactory(DataclassFactory[SimpleValue]):
    __model__ = SimpleValue


def test_extract_financial_values_handles_cached_nested_structure():
    # Setup: create collector instance without running __init__ and a nested cached stmt
    collector = OptimizedFundamentalCollector.__new__(OptimizedFundamentalCollector)

    cached_stmt = {
        "financials": {
            "income_statement": {
                "revenues": {"value": 123.45, "source": "direct_report"}
            },
            "balance_sheet": {
                "assets": {"value": 1000.0, "source": "intra_report_impute"}
            },
        }
    }

    # Execution
    result = collector._extract_financial_values(cached_stmt, ["revenues", "assets"])

    # Verification
    assert result["revenues"] == 123.45
    assert result["assets"] == 1000.0
    # Average of confidences: direct_report->1.0, intra_report_impute->0.8 => (1.0+0.8)/2
    assert pytest.approx(result["data_source_confidence"], rel=1e-3) == 0.9


def test_extract_financial_values_handles_object_format_and_missing_fields():
    # Setup
    collector = OptimizedFundamentalCollector.__new__(OptimizedFundamentalCollector)

    # Create simple object-style statement with attributes
    value_obj = SimpleValueFactory.build()

    class ObjStmt:
        revenues = value_obj
        # assets intentionally missing

    obj_stmt = ObjStmt()

    # Execution
    result = collector._extract_financial_values(obj_stmt, ["revenues", "assets"])

    # Verification
    assert result["revenues"] == value_obj.value
    assert result["assets"] is None
    # Confidence present because value_obj.source may be None
    if value_obj.source:
        assert 0.0 < result["data_source_confidence"] <= 1.0
    else:
        assert result["data_source_confidence"] is None


def test_prepare_raw_data_parses_dates_and_computes_quality():
    # Setup: stub _extract_financial_values to return known values
    collector = OptimizedFundamentalCollector.__new__(OptimizedFundamentalCollector)

    def fake_extract(stmt, fields):
        return {f: (100 if f.endswith("s") else 10) for f in fields}

    collector._extract_financial_values = fake_extract

    income_stmt = {
        "end_date": "2025-01-31",
        "filing_date": "2025-02-15",
        "fiscal_period": "Q1",
        "fiscal_year": 2025,
        "timeframe": "annual",
    }

    # Execution
    raw = collector._prepare_raw_data(42, income_stmt, None, None)

    # Verification
    assert raw["ticker_id"] == 42
    assert isinstance(raw["date"], date)
    assert isinstance(raw["filing_date"], date)
    # Data quality score: fake_extract returns values for income fields only -> non-zero
    assert 0.0 <= raw["data_quality_score"] <= 1.0
    assert isinstance(raw["missing_data_count"], int)


def test_store_statement_period_success_path_calls_execute():
    # Setup
    collector = OptimizedFundamentalCollector.__new__(OptimizedFundamentalCollector)
    collector.ticker_cache = {"TICK": 1}

    income_stmt = {
        "end_date": "2025-01-01",
        "filing_date": "2025-02-01",
        "fiscal_period": "Q1",
        "fiscal_year": 2025,
        "timeframe": "annual",
    }

    response = {"results": [income_stmt]}

    # Stub prepare to return the exact data used by execute placeholders
    collector._prepare_raw_data = lambda ticker_id, inc, bal, cf: {
        "ticker_id": ticker_id,
        "date": inc["end_date"],
        "filing_date": inc["filing_date"],
        "fiscal_period": inc["fiscal_period"],
        "fiscal_year": inc["fiscal_year"],
        "timeframe": inc["timeframe"],
        "revenues": 100.0,
        "cost_of_revenue": None,
        "gross_profit": None,
        "operating_expenses": None,
        "net_income_loss": None,
        "assets": None,
        "current_assets": None,
        "liabilities": None,
        "equity": None,
        "long_term_debt": None,
        "net_cash_flow_from_operating_activities": None,
        "net_cash_flow_from_investing_activities": None,
        "comprehensive_income_loss": None,
        "other_comprehensive_income_loss": None,
        "data_quality_score": 1.0,
        "missing_data_count": 0,
        "source_filing_url": None,
        "source_filing_file_url": None,
        "data_source_confidence": 1.0,
    }

    # Patch execute to ensure it's called and does not raise
    with patch(
        "src.data_collector.polygon_fundamentals.optimized_collector.execute", Mock()
    ) as mock_exec:
        # Execution
        result = asyncio.run(
            collector._store_statement_period(1, income_stmt, response)
        )

        # Verification
        assert result is True
        mock_exec.assert_called()


def test_find_matching_statement_with_dict_and_object_inputs():
    # Setup
    collector = OptimizedFundamentalCollector.__new__(OptimizedFundamentalCollector)

    dict_stmt = {"end_date": "2025-01-01", "fiscal_period": "Q1", "fiscal_year": 2025}

    class ObjStmt:
        def __init__(self):
            self.end_date = "2025-02-02"
            self.fiscal_period = "Q2"
            self.fiscal_year = 2025

    obj = ObjStmt()

    # Execution & Verification for dict match
    assert (
        collector._find_matching_statement([dict_stmt, obj], "2025-01-01", "Q1", 2025)
        == dict_stmt
    )

    # Execution & Verification for object match
    assert (
        collector._find_matching_statement([dict_stmt, obj], "2025-02-02", "Q2", 2025)
        is obj
    )


def test_load_ticker_cache_and_existing_data_cache():
    # Setup
    collector = OptimizedFundamentalCollector.__new__(OptimizedFundamentalCollector)
    collector.data_storage = Mock(
        get_tickers=Mock(
            return_value=[{"ticker": "MSFT", "id": 7}, {"ticker": None, "id": 8}]
        )
    )

    # Execution for ticker cache
    ticker_cache = collector._load_ticker_cache()

    # Verification
    assert ticker_cache == {"MSFT": 7}

    # Setup for existing data cache using patch on fetch_all
    with patch(
        "src.data_collector.polygon_fundamentals.optimized_collector.fetch_all",
        return_value=[
            {"ticker_id": 1, "date": "2025-01-01"},
            {"ticker_id": 2, "date": "2025-02-01"},
        ],
    ):
        # Execution
        collector._load_existing_data_cache()

        # Verification
        assert collector.existing_data_cache == {(1, "2025-01-01"), (2, "2025-02-01")}
