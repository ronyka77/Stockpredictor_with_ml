import pytest

from polyfactory.factories.pydantic_factory import ModelFactory
from src.data_collector.polygon_fundamentals.data_models import (
    FinancialValue,
    IncomeStatement,
    BalanceSheet,
    CashFlowStatement,
)

class FinancialValueFactory(ModelFactory[FinancialValue]):
    __model__ = FinancialValue

class IncomeStatementFactory(ModelFactory[IncomeStatement]):
    __model__ = IncomeStatement

class BalanceSheetFactory(ModelFactory[BalanceSheet]):
    __model__ = BalanceSheet

class CashFlowStatementFactory(ModelFactory[CashFlowStatement]):
    __model__ = CashFlowStatement


@pytest.fixture
def fv_factory():
    if FinancialValueFactory:
        return FinancialValueFactory


@pytest.fixture
def income_factory():
    if IncomeStatementFactory:
        return IncomeStatementFactory


@pytest.fixture
def balance_factory():
    if BalanceSheetFactory:
        return BalanceSheetFactory


@pytest.fixture
def cashflow_factory():
    if CashFlowStatementFactory:
        return CashFlowStatementFactory


