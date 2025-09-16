import time
import pytest
import sys
from pathlib import Path

# Load shared fixtures from tests._fixtures so pytest discovers them
pytest_plugins = [
    "tests._fixtures.conftest",
    "tests._fixtures.fixtures",
]

# Ensure the project `src` package is importable during pytest collection.
# This mirrors editable installs by adding the repository `src/` to sys.path.
root = Path(__file__).resolve().parent
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(autouse=True)
def no_sleep(mocker):
    """Prevent actual sleeping in tests to speed up retry/backoff paths."""
    mocker.patch.object(time, "sleep", lambda s: None)


@pytest.fixture
def patch_execute_values_to_fake_pool(mocker):
    """
    Patch the module-level `execute_values` used in production to delegate inserts to a local fake DB pool.
    
    This fixture creates a ConnectionFake instance and replaces
    `src.data_collector.polygon_data.data_storage.execute_values` with a wrapper that
    calls the fake pool's `execute_values(insert_sql, rows, template, page_size)`.
    The wrapper signature preserves `template`, `page_size`, and `commit` for compatibility;
    the `commit` argument is accepted but ignored by the fake pool. Intended for use
    as a pytest fixture to prevent real database interactions during tests.
    """
    # Create a local fake pool instance (centralized fake implementation)
    from tests._fixtures import ConnectionFake as _FakePool

    fake_pool = _FakePool()

    def _execute_values_wrapper(
        insert_sql, rows, template=None, page_size=1000, commit=True
    ):
        fake_pool.execute_values(
            insert_sql, rows, template=template, page_size=page_size
        )

    mocker.patch(
        "src.data_collector.polygon_data.data_storage.execute_values",
        new=_execute_values_wrapper,
    )
    return None


@pytest.fixture(autouse=True)
def patch_global_db_pool(mocker):
    """
    Patch src.database.connection's global pool helpers to use a test-controlled PoolFake.
    
    This fixture replaces init_global_pool, get_global_pool, and close_global_pool with
    in-process implementations backed by a single PoolFake instance so tests never
    open real database connections. The patched helpers:
    - init_global_pool_fake(minconn=1, maxconn=10): creates and stores a PoolFake.
    - get_global_pool_fake(): returns the stored PoolFake, initializing one if absent.
    - close_global_pool_fake(): clears the stored PoolFake.
    
    The fixture yields once patched so callers (tests) run with the fake pool in place.
    """
    from tests._fixtures import PoolFake

    pool_ref = {"pool": None}

    def init_global_pool_fake(minconn: int = 1, maxconn: int = 10):
        """
        Initialize and store a test fake database connection pool and return it.
        
        Creates a PoolFake instance with the given connection bounds, stores it in the shared
        pool_ref under the "pool" key (replacing any existing value), and returns the instance.
        
        Parameters:
            minconn (int): Minimum number of connections the fake pool should emulate. Defaults to 1.
            maxconn (int): Maximum number of connections the fake pool should emulate. Defaults to 10.
        
        Returns:
            PoolFake: The initialized fake pool instance stored in pool_ref["pool"].
        """
        pool_ref["pool"] = PoolFake(minconn=minconn, maxconn=maxconn)
        return pool_ref["pool"]

    def get_global_pool_fake():
        """
        Return the global test fake database pool, creating and storing one if it does not yet exist.
        
        This function provides a single shared PoolFake instance used by tests; it lazily initializes the pool on first call and subsequently returns the same instance.
        
        Returns:
            PoolFake: The shared fake connection pool used for testing.
        """
        if pool_ref["pool"] is None:
            return init_global_pool_fake()
        return pool_ref["pool"]

    def close_global_pool_fake():
        pool_ref["pool"] = None

    mocker.patch("src.database.connection.init_global_pool", init_global_pool_fake)
    mocker.patch("src.database.connection.get_global_pool", get_global_pool_fake)
    mocker.patch("src.database.connection.close_global_pool", close_global_pool_fake)
    yield


@pytest.fixture(autouse=True)
def patch_polygon_api_clients(mocker):
    """Autouse fixture: prevent any test from performing real network calls"""
    # Use canonical canned response factory from fixtures
    from tests._fixtures.remote_api_responses import canned_api_factory

    # Default network behavior: patch Session.get to return an empty canned
    mocker.patch(
        "requests.Session.get",
        return_value=canned_api_factory("empty"),
    )

    # Default news client pagination fetcher -> empty list
    mocker.patch(
        "src.data_collector.polygon_news.news_client.PolygonNewsClient._fetch_paginated_data",
        return_value=[],
    )

    yield
