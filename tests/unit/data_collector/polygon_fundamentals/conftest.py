import pytest
from pathlib import Path

try:
    from polyfactory.factories.pydantic_factory import ModelFactory
    from src.data_collector.polygon_fundamentals.data_models import FinancialValue

    class FinancialValueFactory(ModelFactory[FinancialValue]):
        __model__ = FinancialValue

except Exception:  # pragma: no cover - optional dependency
    FinancialValueFactory = None
    from src.data_collector.polygon_fundamentals.data_models import FinancialValue


@pytest.fixture
def fv_factory():
    if FinancialValueFactory:
        return FinancialValueFactory

    class _SimpleFV:
        @staticmethod
        def build(**kwargs):
            return FinancialValue(**{**{"value": kwargs.get("value", 1.0)}, **kwargs})

    return _SimpleFV


