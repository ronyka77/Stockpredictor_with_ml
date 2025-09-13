import pytest

from src.data_collector.polygon_fundamentals_v2.parser import FundamentalsParser


def test_parser_returns_empty_response_for_no_results():
    parser = FundamentalsParser()
    payload = {"status": "OK", "results": []}

    dto = parser.parse(payload, "TST")

    assert dto.income_statements == []
    assert dto.balance_sheets == []
    assert dto.cash_flow_statements == []
    assert dto.calculate_data_quality() == 0.0


def test_parser_handles_partial_missing_fields():
    parser = FundamentalsParser()
    payload = {
        "status": "OK",
        "results": [
            {"end_date": "2024-03-31", "revenues": None, "assets": 1000}
        ],
    }

    dto = parser.parse(payload, "TST")

    # Partial fields should still produce statements but quality is low
    assert dto.income_statements or dto.balance_sheets
    assert 0.0 <= dto.data_quality_score <= 1.0


