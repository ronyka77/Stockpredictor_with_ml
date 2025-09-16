import pytest

from polyfactory.factories.pydantic_factory import ModelFactory
from src.data_collector.polygon_fundamentals.data_models import (
    FinancialValue,
)


class FinancialValueFactory(ModelFactory[FinancialValue]):
    __model__ = FinancialValue


@pytest.fixture
def fv_factory():
    if FinancialValueFactory:
        return FinancialValueFactory

