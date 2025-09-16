import pytest

from polyfactory.factories.pydantic_factory import ModelFactory
from src.data_collector.polygon_fundamentals.data_models import FinancialValue


class FinancialValueFactory(ModelFactory[FinancialValue]):
    __model__ = FinancialValue


@pytest.fixture
def fv_factory():
    """
    Pytest fixture that provides the FinancialValue factory class for tests.
    
    Returns the FinancialValueFactory class used to construct FinancialValue instances in tests. If the factory is not set (falsy), the fixture returns None.
    """
    if FinancialValueFactory:
        return FinancialValueFactory
