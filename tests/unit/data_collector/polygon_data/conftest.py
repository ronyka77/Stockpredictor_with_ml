import datetime
import pytest
from unittest.mock import patch

from polyfactory.factories.pydantic_factory import ModelFactory
from src.data_collector.polygon_data.data_validator import OHLCVRecord
from src.data_collector.polygon_data.client import PolygonDataClient


# Shared PolygonDataClient fixture
@pytest.fixture
def polygon_client():
    """Return a configured PolygonDataClient for tests."""
    return PolygonDataClient(api_key="TEST", requests_per_minute=100)


# Simple helper to patch object methods with autospec in tests
@pytest.fixture
def patch_object_autospec():
    """
    Pytest fixture that returns a helper wrapper around unittest.mock.patch.object which enforces autospec=True.
    
    The returned callable has the signature (target, attribute, **kwargs) and behaves like unittest.mock.patch.object,
    automatically setting autospec=True. Use it as a context manager or decorator in tests, for example:
    
    with patch_object_autospec(SomeClass, "method", return_value=...):
        ...
    """

    def _wrapper(target, attribute, **kwargs):
        """
        Create a unittest.mock.patch.object wrapper that enforces autospec=True.
        
        Parameters:
            target: The object or module whose attribute will be patched.
            attribute (str): Name of the attribute to patch on the target.
            **kwargs: Any additional keyword arguments forwarded to unittest.mock.patch.object.
        
        Returns:
            A patcher (as returned by unittest.mock.patch.object) configured with autospec=True.
        """
        return patch.object(target, attribute, autospec=True, **kwargs)

    return _wrapper


class OHLCVRecordFactory(ModelFactory):
    __model__ = OHLCVRecord

    # sensible defaults for tests; override in calls
    ticker = "AAPL"
    timestamp = datetime.date(2020, 1, 1)
    open = 9.0
    high = 11.0
    low = 8.0
    close = 10.0
    volume = 1000


@pytest.fixture
def poly_ohlcv_factory():
    """
    Return the OHLCVRecordFactory class for creating test OHLCVRecord instances.
    
    Use the returned factory to build records in tests, e.g. `OHLCVRecordFactory.build()` or `OHLCVRecordFactory.batch(n)`.
    """
    return OHLCVRecordFactory
