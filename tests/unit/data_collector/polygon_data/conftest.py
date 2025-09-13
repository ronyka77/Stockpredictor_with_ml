import datetime
import pytest
from unittest.mock import MagicMock, patch

from polyfactory.factories.pydantic_factory import ModelFactory
from src.data_collector.polygon_data.data_validator import OHLCVRecord
from src.data_collector.polygon_data.client import PolygonDataClient

# Lightweight fake response factory to replace ad-hoc resp._payload usage
@pytest.fixture
def fake_response():
	def _make(payload=None, status=200):
		resp = MagicMock()
		resp.status_code = status
		resp.json.return_value = payload or {}
		return resp
	return _make

# Shared PolygonDataClient fixture
@pytest.fixture
def polygon_client():
	"""Return a configured PolygonDataClient for tests."""
	return PolygonDataClient(api_key="TEST", requests_per_minute=100)

# Reusable simple OHLCV record builder (use polyfactory factories when available)
@pytest.fixture
def ohlcv_record():
	def _make(ticker: str = "AAPL", dt: datetime.date = None, close: float = 10.0):
		if dt is None:
			dt = datetime.date(2020, 1, 1)
		return {
			"ticker": ticker,
			"timestamp": dt,
			"open": close - 1.0,
			"high": close + 1.0,
			"low": close - 2.0,
			"close": close,
			"volume": 1000,
		}
	return _make

# Simple helper to patch object methods with autospec in tests
@pytest.fixture
def patch_object_autospec():
	"""Yield a helper wrapper around unittest.mock.patch.object that sets autospec=True.

	Usage in tests:
	with patch_object_autospec(SomeClass, 'method', return_value=...):
		...
	"""
	def _wrapper(target, attribute, **kwargs):
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
    """Return the OHLCVRecordFactory for use in tests: OHLCVRecordFactory.build()

    If polyfactory is not available, tests can still use `ohlcv_record` above.
    """
    return OHLCVRecordFactory
