import time
import pytest
import sys
from pathlib import Path

# Ensure the project `src` package is importable during pytest collection.
# This mirrors editable installs by adding the repository `src/` to sys.path.
root = Path(__file__).resolve().parent
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tests._fixtures.helpers import make_sample_df, make_fake_http_response, FakePool


@pytest.fixture
def sample_df():
    return make_sample_df()


@pytest.fixture
def fake_response_factory():
    return make_fake_http_response


@pytest.fixture
def fake_pool():
    return FakePool()


@pytest.fixture(autouse=True)
def _patch_threaded_connection_pool(mocker):
    """Autouse safety: prevent tests from creating real ThreadedConnectionPool.

    Patches `src.database.connection.ThreadedConnectionPool` to use the fake threaded
    pool implementation from test helpers.
    """
    from tests._fixtures.helpers import FakeThreadedPool

    mocker.patch("src.database.connection.ThreadedConnectionPool", FakeThreadedPool)
    yield


@pytest.fixture(autouse=True)
def no_sleep(mocker):
    """Prevent actual sleeping in tests to speed up retry/backoff paths."""
    mocker.patch.object(time, "sleep", lambda s: None)


@pytest.fixture
def patch_execute_values_to_fake_pool(mocker, fake_pool):
    """Expose wrapper fixture at top-level tests for delegating execute_values to fake pool."""

    def _execute_values_wrapper(insert_sql, rows, template=None, page_size=1000, commit=True):
        fake_pool.execute_values(insert_sql, rows, template=template, page_size=page_size)

    mocker.patch(
        "src.data_collector.polygon_data.data_storage.execute_values",
        new=_execute_values_wrapper,
    )
    return None
