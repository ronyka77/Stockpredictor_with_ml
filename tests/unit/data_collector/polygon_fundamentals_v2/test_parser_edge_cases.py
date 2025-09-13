from src.data_collector.polygon_fundamentals_v2.parser import FundamentalsParser
from src.data_collector.polygon_fundamentals.data_models import FundamentalDataResponse


def test_parser_handles_missing_results():
    parser = FundamentalsParser()
    raw = {"status": "OK"}
    resp = parser.parse(raw, "AAPL")
    assert isinstance(resp, FundamentalDataResponse)
    assert resp.income_statements == []


def test_parser_prefers_nested_over_legacy_and_parses_values():
    parser = FundamentalsParser()
    raw = {
        "results": [
            {
                "start_date": "2023-01-01",
                "end_date": "2023-03-31",
                "financials": {
                    "income_statement": {"revenues": {"value": 100}}
                },
            }
        ]
    }

    resp = parser.parse(raw, "AAPL")
    assert resp.income_statements
    assert resp.income_statements[0].revenues.value == 100


