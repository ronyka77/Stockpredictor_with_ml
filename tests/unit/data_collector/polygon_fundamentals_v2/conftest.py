import pytest

from polyfactory.factories.pydantic_factory import ModelFactory
from src.data_collector.polygon_fundamentals.data_models import (
    FinancialValue,
)


class FinancialValueFactory(ModelFactory[FinancialValue]):
    __model__ = FinancialValue


@pytest.fixture
def fv_factory():
    """
    Pytest fixture that provides the FinancialValueFactory class for tests.
    
    Returns the FinancialValueFactory class if it's available in the module; otherwise returns None. Tests use this fixture to construct FinancialValue model instances.
    """
    if FinancialValueFactory:
        return FinancialValueFactory

