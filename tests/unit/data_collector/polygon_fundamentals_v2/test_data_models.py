from datetime import date
import pytest
import numpy as np

from src.data_collector.polygon_fundamentals import data_models as dm


def make_financial_value(v=100.0):
    return dm.FinancialValue(
        value=float(v), unit="USD", label="test", order=1, source="polygon"
    )


def test_extract_financial_value_dict_and_numeric_and_missing():
    """Extract numeric values from nested dicts and handle missing keys"""
    data = {"revenues": {"value": 123.0}, "net": 50}
    if not np.isclose(dm.extract_financial_value(data, "revenues"), 123.0):
        raise AssertionError("extract_financial_value failed for revenues")
    if not np.isclose(dm.extract_financial_value(data, "net"), 50.0):
        raise AssertionError("extract_financial_value failed for net")
    if dm.extract_financial_value(data, "missing") is not None:
        raise AssertionError(
            "extract_financial_value failed to return None for missing"
        )


def test_safe_divide_and_growth_and_cagr():
    """Test safe division, growth rate, and CAGR calculations including edge cases"""
    # safe_divide
    if dm.safe_divide(10, 2) != 5:
        raise AssertionError("safe_divide returned unexpected result")
    if dm.safe_divide(None, 2) is not None:
        raise AssertionError("safe_divide did not return None for None numerator")
    if dm.safe_divide(1, 0) is not None:
        raise AssertionError("safe_divide did not return None for division by zero")

    # growth rate
    if dm.calculate_growth_rate(120, 100) != pytest.approx(0.2):
        raise AssertionError("calculate_growth_rate returned unexpected value")
    if dm.calculate_growth_rate(None, 100) is not None:
        raise AssertionError("calculate_growth_rate did not return None for None input")
    if dm.calculate_growth_rate(100, 0) is not None:
        raise AssertionError(
            "calculate_growth_rate did not return None when denominator is zero"
        )

    # cagr
    if dm.calculate_cagr(200, 100, 2) != pytest.approx((200 / 100) ** (1 / 2) - 1):
        raise AssertionError("calculate_cagr returned unexpected value")
    if dm.calculate_cagr(None, 100, 2) is not None:
        raise AssertionError("calculate_cagr did not return None for None input")
    if dm.calculate_cagr(200, 0, 2) is not None:
        raise AssertionError("calculate_cagr did not return None for zero denominator")
    if dm.calculate_cagr(200, 100, 0) is not None:
        raise AssertionError("calculate_cagr did not return None for zero periods")


def test_financial_statement_date_parsing_and_company_details():
    """Parse date strings into date objects for statements and company details"""
    # Create statements with string dates to ensure field_validator parses them
    inc = dm.IncomeStatement(
        start_date="2020-01-01", filing_date="2021-03-01", fiscal_period="Q1"
    )
    dm.BalanceSheet(filing_date="2020-12-31", fiscal_period="FY")
    dm.CashFlowStatement(filing_date="2021-06-30", fiscal_period="Q2")

    if not isinstance(inc.start_date, date):
        raise AssertionError("IncomeStatement start_date not parsed to date")
    if not isinstance(inc.filing_date, date):
        raise AssertionError("IncomeStatement filing_date not parsed to date")
    if inc.fiscal_period != "Q1":
        raise AssertionError("IncomeStatement fiscal_period mismatch")

    # CompanyDetails parsing list_date
    comp = dm.CompanyDetails(ticker="TICK", list_date="2010-05-01")
    if not isinstance(comp.list_date, date):
        raise AssertionError("CompanyDetails list_date not parsed to date")


def test_fundamental_data_response_latest_and_by_period_and_quality():
    """Validate selection of latest statements and calculation of data quality"""
    # Build several statements with filing_dates
    s1 = dm.IncomeStatement(filing_date="2020-01-01", fiscal_period="Q1")
    s2 = dm.IncomeStatement(filing_date="2021-01-01", fiscal_period="Q1")
    b1 = dm.BalanceSheet(filing_date="2020-01-01", fiscal_period="Q1")
    cf1 = dm.CashFlowStatement(filing_date="2019-12-31", fiscal_period="FY")

    # Populate some essential fields for completeness calculation
    s1.revenues = make_financial_value(100)
    s1.net_income_loss = make_financial_value(10)
    # s1 missing earnings_per_share_basic etc.

    s2.revenues = make_financial_value(200)
    s2.net_income_loss = make_financial_value(20)
    s2.earnings_per_share_basic = make_financial_value(1.5)
    s2.weighted_average_shares_outstanding = make_financial_value(1000)
    s2.operating_income_loss = make_financial_value(15)

    b1.assets = make_financial_value(500)
    b1.equity = make_financial_value(300)
    b1.current_assets = make_financial_value(200)
    b1.current_liabilities = make_financial_value(50)
    b1.liabilities = make_financial_value(200)

    cf1.net_cash_flow_from_operating_activities = make_financial_value(30)

    resp = dm.FundamentalDataResponse(
        status="OK",
        income_statements=[s1, s2],
        balance_sheets=[b1],
        cash_flow_statements=[cf1],
    )

    # Latest income statement should be s2 (2021)
    latest_inc = resp.get_latest_income_statement()
    if latest_inc is not s2:
        raise AssertionError("Latest income statement selection incorrect")

    # Latest balance sheet
    latest_bal = resp.get_latest_balance_sheet()
    if latest_bal is not b1:
        raise AssertionError("Latest balance sheet selection incorrect")

    # get statements by period
    by_q1 = resp.get_statements_by_period("Q1")
    if by_q1["income_statement"] is None:
        raise AssertionError("get_statements_by_period missing income_statement")
    if by_q1["balance_sheet"] is None:
        raise AssertionError("get_statements_by_period missing balance_sheet")

    # calculate data quality
    score = resp.calculate_data_quality()
    if not isinstance(score, float):
        raise AssertionError("Data quality score type mismatch")
    # Since some essential fields are provided, score should be > 0
    if score <= 0:
        raise AssertionError("Data quality score unexpectedly non-positive")
