import time
import pytest
import sys
from pathlib import Path

# Ensure the project `src` package is importable during pytest collection.
# This mirrors editable installs by adding the repository `src/` to sys.path.
root = Path(__file__).resolve().parent.parent
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


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
    """Return a helper that mocks filesystem calls to raise PermissionError."""

    def _apply(mocker):
        import os
        import builtins

        def _raise(*a, **k):
            raise PermissionError("Permission denied (simulated)")

        # Patch os.makedirs and builtins.open to raise PermissionError
        mocker.patch.object(os, "makedirs", _raise)
        mocker.patch.object(builtins, "open", _raise)

    return _apply


@pytest.fixture(autouse=True)
def no_sleep(mocker):
    """Prevent actual sleeping in tests to speed up retry/backoff paths."""
    mocker.patch.object(time, "sleep", lambda s: None)
