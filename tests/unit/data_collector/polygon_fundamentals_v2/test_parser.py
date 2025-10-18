import pytest
import numpy as np

from src.data_collector.polygon_fundamentals_v2.parser import FundamentalsParser
from src.data_collector.polygon_fundamentals.data_models import (
    FundamentalDataResponse,
)


@pytest.fixture()
def parser() -> FundamentalsParser:
    return FundamentalsParser()


@pytest.mark.parametrize(
    "format_type, expected_revenue",
    [("nested", 100.0), ("legacy", 200.0)],
)
def test_parser_handles_nested_and_legacy_formats(
    parser, fv_factory, format_type, expected_revenue
):
    """Parse both nested and legacy payload formats and extract revenues"""
    # Setup
    if format_type == "nested":
        fv = fv_factory().build(value=expected_revenue)
        # pydantic models produced by factories expose model_dump(); use it for JSON-like dict
        revenues_field = fv.model_dump() if hasattr(fv, "model_dump") else fv.__dict__
        payload = {
            "status": "OK",
            "results": [
                {
                    "start_date": "2024-01-01",
                    "end_date": "2024-03-31",
                    "filing_date": "2024-05-01",
                    "timeframe": "quarterly",
                    "fiscal_period": "Q1",
                    "fiscal_year": "2024",
                    "company_name": "Test Co",
                    "financials": {
                        "income_statement": {
                            "revenues": revenues_field,
                            "net_income_loss": {"value": 10.0},
                        },
                        "balance_sheet": {"assets": {"value": 1000.0}},
                        "cash_flow_statement": {"net_cash_flow": {"value": 5.0}},
                    },
                }
            ],
        }
    else:
        payload = {
            "status": "OK",
            "results": [
                {
                    "start_date": "2024-01-01",
                    "end_date": "2024-03-31",
                    "filing_date": "2024-05-01",
                    "timeframe": "quarterly",
                    "fiscal_period": "Q1",
                    "fiscal_year": "2024",
                    "company_name": "Legacy Co",
                    "revenues": expected_revenue,
                    "net_income_loss": 20.0,
                    "assets": 2000.0,
                    "net_cash_flow": 15.0,
                }
            ],
        }

    # Execution
    dto = parser.parse(payload, "TST")

    # Verification
    assert isinstance(dto, FundamentalDataResponse)
    assert dto.income_statements, "income_statements should be present"
    assert dto.balance_sheets, "balance_sheets should be present"
    assert dto.cash_flow_statements, "cash_flow_statements should be present"

    latest_inc = dto.get_latest_income_statement()
    assert latest_inc is not None
    assert latest_inc.revenues.value == expected_revenue


def test_parser_handles_missing_results():
    """Return empty FundamentalDataResponse when results key is missing"""
    parser = FundamentalsParser()
    raw = {"status": "OK"}
    resp = parser.parse(raw, "AAPL")
    assert isinstance(resp, FundamentalDataResponse)
    assert resp.income_statements == []


def test_parser_prefers_nested_over_legacy_and_parses_values():
    """Prefer nested financials format and correctly extract revenue values"""
    parser = FundamentalsParser()
    raw = {
        "results": [
            {
                "start_date": "2023-01-01",
                "end_date": "2023-03-31",
                "financials": {"income_statement": {"revenues": {"value": 100}}},
            }
        ]
    }

    resp = parser.parse(raw, "AAPL")
    assert resp.income_statements
    assert resp.income_statements[0].revenues.value == 100


def test_parser_returns_empty_response_for_no_results():
    """Return empty DTO when parser receives no results in payload"""
    parser = FundamentalsParser()
    payload = {"status": "OK", "results": []}

    dto = parser.parse(payload, "TST")

    assert dto.income_statements == []
    assert dto.balance_sheets == []
    assert dto.cash_flow_statements == []
    assert np.isclose(dto.calculate_data_quality(), 0.0)


def test_parser_handles_partial_missing_fields():
    """Handle payloads with partial/missing financial fields gracefully"""
    parser = FundamentalsParser()
    payload = {
        "status": "OK",
        "results": [{"end_date": "2024-03-31", "revenues": None, "assets": 1000}],
    }

    dto = parser.parse(payload, "TST")

    # Partial fields should still produce statements but quality is low
    assert dto.income_statements or dto.balance_sheets
    assert 0.0 <= dto.data_quality_score <= 1.0
