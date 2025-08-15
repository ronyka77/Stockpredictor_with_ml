import pytest
from types import SimpleNamespace


@pytest.fixture
def mock_http_client():
    """Simple mock HTTP client fixture with configurable return values."""
    class MockClient:
        def __init__(self):
            self._aggregates = None
            self._grouped = None

        def set_aggregates(self, data):
            self._aggregates = data

        def set_grouped(self, data):
            self._grouped = data

        def get_aggregates(self, **kwargs):
            if isinstance(self._aggregates, Exception):
                raise self._aggregates
            return self._aggregates

        def get_grouped_daily(self, date_str):
            if isinstance(self._grouped, Exception):
                raise self._grouped
            return self._grouped

    return MockClient()


@pytest.fixture
def permission_error_simulator():
    """Return a helper that monkeypatches filesystem calls to raise PermissionError."""
    def _apply(monkeypatch):
        import os
        import builtins

        def _raise(*a, **k):
            raise PermissionError("Permission denied (simulated)")

        monkeypatch.setattr(os, "makedirs", _raise)
        monkeypatch.setattr(builtins, "open", _raise)

    return _apply


