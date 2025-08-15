from datetime import date
import pytest

from src.data_collector.polygon_fundamentals import data_models as dm


def make_financial_value(v=100.0):
    return dm.FinancialValue(value=float(v), unit="USD", label="test", order=1, source="polygon")


def test_extract_financial_value_dict_and_numeric_and_missing():
    data = {"revenues": {"value": 123.0}, "net": 50}
    assert dm.extract_financial_value(data, "revenues") == 123.0
    assert dm.extract_financial_value(data, "net") == 50.0
    assert dm.extract_financial_value(data, "missing") is None


def test_safe_divide_and_growth_and_cagr():
    # safe_divide
    assert dm.safe_divide(10, 2) == 5
    assert dm.safe_divide(None, 2) is None
    assert dm.safe_divide(1, 0) is None

    # growth rate
    assert dm.calculate_growth_rate(120, 100) == pytest.approx(0.2)
    assert dm.calculate_growth_rate(None, 100) is None
    assert dm.calculate_growth_rate(100, 0) is None

    # cagr
    assert dm.calculate_cagr(200, 100, 2) == pytest.approx((200 / 100) ** (1 / 2) - 1)
    assert dm.calculate_cagr(None, 100, 2) is None
    assert dm.calculate_cagr(200, 0, 2) is None
    assert dm.calculate_cagr(200, 100, 0) is None


def test_financial_statement_date_parsing_and_company_details():
    # Create statements with string dates to ensure field_validator parses them
    inc = dm.IncomeStatement(start_date="2020-01-01", filing_date="2021-03-01", fiscal_period="Q1")
    bal = dm.BalanceSheet(filing_date="2020-12-31", fiscal_period="FY")
    cf = dm.CashFlowStatement(filing_date="2021-06-30", fiscal_period="Q2")

    assert isinstance(inc.start_date, date)
    assert isinstance(inc.filing_date, date)
    assert inc.fiscal_period == "Q1"

    # CompanyDetails parsing list_date
    comp = dm.CompanyDetails(ticker="TICK", list_date="2010-05-01")
    assert isinstance(comp.list_date, date)


def test_fundamental_data_response_latest_and_by_period_and_quality():
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
        cash_flow_statements=[cf1]
    )

    # Latest income statement should be s2 (2021)
    latest_inc = resp.get_latest_income_statement()
    assert latest_inc is s2

    # Latest balance sheet
    latest_bal = resp.get_latest_balance_sheet()
    assert latest_bal is b1

    # get statements by period
    by_q1 = resp.get_statements_by_period("Q1")
    assert by_q1["income_statement"] is not None
    assert by_q1["balance_sheet"] is not None

    # calculate data quality
    score = resp.calculate_data_quality()
    assert isinstance(score, float)
    # Since some essential fields are provided, score should be > 0
    assert score > 0


