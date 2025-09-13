import os
import builtins
import pytest


# Central deterministic seed fixture for all tests (Polyfactory + numeric libs)
@pytest.fixture(scope="session", autouse=True)
def factory_seed():
    """Set a single deterministic seed for factories and numeric RNGs.

    This fixture runs once per test session and makes Polyfactory-generated
    data and any numpy/random-based behavior deterministic in CI.
    """
    seed = 42

    # Seed Python random
    import random

    random.seed(seed)

    # Seed numpy if available
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    return seed


@pytest.fixture
def mock_http_client():
    """Simple mock HTTP client fixture with configurable return values.

    Maintains `set_aggregates` and `set_grouped` to allow tests to inject
    deterministic responses or exceptions.
    """

    class MockClient:
        def __init__(self):
            self._aggregates = None
            self._grouped = None

        # def set_aggregates(self, data):
        #     self._aggregates = data

        # def set_grouped(self, data):
        #     self._grouped = data

        def get_aggregates(self, *args, **kwargs):
            if isinstance(self._aggregates, Exception):
                raise self._aggregates
            return self._aggregates

        def get_grouped_daily(self, *args, **kwargs):
            if isinstance(self._grouped, Exception):
                raise self._grouped
            return self._grouped

    return MockClient()


@pytest.fixture
def make_fake_http_response():
    """Factory fixture returning a function that constructs fake HTTP responses.

    Returns a callable `make_fake_http_response(status, json_data, raise_on_json)`.
    """

    def _factory(status=200, json_data=None, raise_on_json=False):
        class FakeResponse:
            def __init__(self, status, payload, raise_on_json):
                self.status_code = status
                self._payload = payload
                self._raise = raise_on_json

            def json(self):
                if self._raise:
                    raise ValueError("malformed json")
                return self._payload

            # def raise_for_status(self):
            #     if not (200 <= self.status_code < 300):
            #         raise Exception(f"HTTP {self.status_code}")

        return FakeResponse(status, json_data, raise_on_json)

    return _factory


@pytest.fixture
def fake_http_response(make_fake_http_response):
    """Alias fixture for tests that request fake_http_response."""
    return make_fake_http_response


@pytest.fixture
def permission_error_simulator():
    """Return a helper that mocks filesystem calls to raise PermissionError."""

    def _apply(mocker):
        def _raise(*a, **k):
            raise PermissionError("Permission denied (simulated)")

        mocker.patch.object(os, "makedirs", _raise)
        mocker.patch.object(builtins, "open", _raise)

    return _apply
