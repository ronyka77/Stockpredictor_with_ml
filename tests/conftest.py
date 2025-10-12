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
    """Expose wrapper fixture at top-level tests for delegating execute_values to fake pool."""
    # Create a local fake pool instance (centralized fake implementation)
    from tests._fixtures import ConnectionFake as _FakePool

    fake_pool = _FakePool()

    def _execute_values_wrapper(
        insert_sql, rows, **kwargs
    ):
        fake_pool.execute_values(
            insert_sql, rows
        )

    mocker.patch(
        "src.data_collector.polygon_data.data_storage.execute_values",
        new=_execute_values_wrapper,
    )
    return None


@pytest.fixture(autouse=True)
def patch_global_db_pool(mocker):
    """Patch global pool helpers to use a fake Postgres-like pool from test helpers.

    Ensures `init_global_pool`, `get_global_pool`, and `close_global_pool` in
    `src.database.connection` operate against a test-controlled fake pool so
    tests never touch a real database connection.
    """
    from tests._fixtures import PoolFake

    pool_ref = {"pool": None}

    def init_global_pool_fake(minconn: int = 1, maxconn: int = 10):
        pool_ref["pool"] = PoolFake(minconn=minconn, maxconn=maxconn)
        return pool_ref["pool"]

    def get_global_pool_fake():
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
