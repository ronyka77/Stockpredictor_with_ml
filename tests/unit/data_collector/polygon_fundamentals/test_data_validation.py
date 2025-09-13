from src.data_collector.polygon_fundamentals.data_validator import (
    FundamentalDataValidator,
    ValidationResult,
)
from src.data_collector.polygon_fundamentals.data_models import (
    FundamentalDataResponse,
    IncomeStatement,
    FinancialValue,
)


def make_income_stmt(revenues=None, gross=None, cost=None):
    return IncomeStatement(
        filing_date="2023-01-01",
        revenues=FinancialValue(value=revenues) if revenues is not None else None,
        gross_profit=FinancialValue(value=gross) if gross is not None else None,
        cost_of_revenue=FinancialValue(value=cost) if cost is not None else None,
    )


def test_validate_response_no_statements():
    validator = FundamentalDataValidator()
    resp = FundamentalDataResponse(status="OK")
    result = validator.validate_response(resp)
    assert result.is_valid is False
    assert "No financial statements found" in result.errors


def test_validate_income_statement_warnings_and_outliers():
    validator = FundamentalDataValidator()
    # revenues negative should trigger warning; extremely high revenues trigger outlier
    stmt1 = make_income_stmt(revenues=-100, gross=50, cost=60)
    stmt2 = make_income_stmt(revenues=2e13, gross=1e13, cost=1e13)

    resp = FundamentalDataResponse(status="OK", income_statements=[stmt1, stmt2])
    result = validator.validate_response(resp)

    # validator reports warnings and outliers and records missing essential fields
    assert "Negative revenues in income statement" in result.warnings
    assert "Extremely high revenues" in result.outliers
    assert any("income_statement.earnings_per_share_basic" in mf for mf in result.missing_fields)


