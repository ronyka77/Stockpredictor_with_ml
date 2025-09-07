import os
import builtins
import time
import pytest
from tests._fixtures.helpers import make_sample_df, make_fake_db, make_fake_http_response


@pytest.fixture
def fake_http_response():
    """Return a factory for fake HTTP responses used across client tests.

    Usage:
        resp = fake_http_response(status=200, json_data={...})
        mocker.patch.object(client.session, 'get', return_value=resp)
    """
    def _factory(status=200, json_data=None, raise_on_json=False):
        return make_fake_http_response(status=status, json_data=json_data, raise_on_json=raise_on_json)

    return _factory


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

        def set_aggregates(self, data):
            self._aggregates = data

        def set_grouped(self, data):
            self._grouped = data

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
def patch_execute_values_to_fake_pool(mocker, fake_pool):
    """Patch the low-level execute_values to delegate to the shared fake pool.

    Usage: include `patch_execute_values_to_fake_pool` in the test signature and
    the fixture will replace `src.data_collector.polygon_data.data_storage.execute_values`
    with a wrapper that calls `fake_pool.execute_values`.
    """

    def _execute_values_wrapper(insert_sql, rows, template=None, page_size=1000, commit=True):
        # delegate to shared fake pool implementation
        fake_pool.execute_values(insert_sql, rows, template=template, page_size=page_size)

    mocker.patch(
        "src.data_collector.polygon_data.data_storage.execute_values",
        new=_execute_values_wrapper,
    )
    return None


@pytest.fixture
def make_sample_dataframe():
    return make_sample_df


@pytest.fixture
def fake_pool():
    return make_fake_db("logical")


@pytest.fixture
def permission_error_simulator():
    """Return a helper that mocks filesystem calls to raise PermissionError."""

    def _apply(mocker):
        def _raise(*a, **k):
            raise PermissionError("Permission denied (simulated)")

        mocker.patch.object(os, "makedirs", _raise)
        mocker.patch.object(builtins, "open", _raise)

    return _apply
