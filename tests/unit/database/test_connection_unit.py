import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
import os

from src.database.connection import (
    PostgresConnection,
    run_in_transaction,
    get_global_pool,
    init_global_pool,
    close_global_pool,
)


def test_databaseconnection_raises_when_password_missing():
    cfg = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": ""}
    with pytest.raises(ValueError) as exc:
        # pool init should validate missing password
        init_kwargs = cfg.copy()
        PostgresConnection(1, 1, **init_kwargs)
    assert "DB_PASSWORD" in str(exc.value), (
        f"Expected missing password error; got: {exc.value}"
    )


def test_safe_pooled_connection_timeout_and_release():
    fake_kwargs = {
        "host": "h",
        "port": 5432,
        "database": "d",
        "user": "u",
        "password": "p",
    }
    # Patch ThreadedConnectionPool to avoid real DB network calls during init
    with patch("src.database.connection.ThreadedConnectionPool"):
        sp = PostgresConnection(1, 1, **fake_kwargs)

    # Acquire internal semaphore to simulate exhaustion
    acquired = sp._sem.acquire(timeout=0)
    try:
        with pytest.raises(RuntimeError):
            with sp.connection(timeout=0.01):
                pass
    finally:
        if acquired:
            sp._sem.release()


def test_get_connection_pool_singleton(mocker):
    # Use the canonical PoolFake via patch_global_pool
    from tests._fixtures.db import patch_global_pool, PoolFake

    pool = PoolFake()
    patch_global_pool(mocker, pool)

    p1 = get_global_pool()
    p2 = get_global_pool()
    if p1 is not p2:
        raise AssertionError(
            "get_connection_pool should return a singleton instance for same params"
        )


def test_get_connection_pool_reinitializes_on_param_change(mocker):
    # Use init_global_pool explicitly to avoid potential interference from
    # other tests that may have patched get_global_pool. Patch ThreadedConnectionPool
    # and ensure required env is present.
    with patch.dict(os.environ, {"DB_PASSWORD": "p"}):
        with patch("src.database.connection.ThreadedConnectionPool"):
            close_global_pool()
            p1 = init_global_pool(minconn=1, maxconn=1)
            close_global_pool()
            p2 = init_global_pool(minconn=1, maxconn=1)
            assert p1 is not p2


def test_run_in_transaction_commits_and_rolls_back_on_exception(mocker):
    # Create a dummy pool that yields a fake connection
    fake_conn = MagicMock()
    fake_cursor = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    @contextmanager
    def conn_cm():
        yield fake_conn

    class DummyPool:
        def connection(self):
            return conn_cm()

    from tests._fixtures.db import patch_global_pool

    patch_global_pool(mocker, DummyPool())

    # Success path
    def success_fn(conn, cur):
        return "ok"

    res = run_in_transaction(success_fn)
    assert res == "ok", "run_in_transaction should return the callable's result"
    assert fake_conn.commit.called, "Expected commit to be called on success"

    # Failure path
    def fail_fn(conn, cur):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_in_transaction(fail_fn)
    assert fake_conn.rollback.called, "Expected rollback to be called on exception"


@pytest.fixture
def patched_threaded_pool():
    """Patch ThreadedConnectionPool to return a simple MagicMock pool.

    Use this fixture when creating PostgresConnection to avoid network I/O.
    """
    with patch("src.database.connection.ThreadedConnectionPool") as pool_cls:
        fake_pool = MagicMock()
        pool_cls.return_value = fake_pool
        yield fake_pool


def test_connection_replaces_broken_conn_and_calls_putconn_close(patched_threaded_pool):
    # Create a PostgresConnection object but inject a fake pool that returns
    # a broken connection first, then a healthy one.
    broken_conn = MagicMock()
    broken_conn.closed = 1
    good_conn = MagicMock()
    good_conn.closed = 0

    # Configure fake pool behavior
    patched_threaded_pool.getconn.side_effect = [broken_conn, good_conn]

    pc = PostgresConnection.__new__(PostgresConnection)
    pc._minconn = 1
    pc._maxconn = 2
    pc._pool = patched_threaded_pool
    from threading import Semaphore

    pc._sem = Semaphore(2)
    pc._closed = False

    with pc.connection() as conn:
        assert conn is good_conn

    # Ensure the broken connection was returned with close=True
    patched_threaded_pool.putconn.assert_any_call(broken_conn, close=True)
    # Ensure the good connection was put back normally
    patched_threaded_pool.putconn.assert_any_call(good_conn)


def test_fetch_and_execute_helpers_use_global_pool():
    # Build a dummy pool that yields a fake connection and cursor
    fake_conn = MagicMock()
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = [{"a": 1}]
    fake_cursor.fetchone.return_value = {"a": 1}
    fake_conn.cursor.return_value = fake_cursor

    @contextmanager
    def conn_cm():
        yield fake_conn

    class DummyPool:
        def connection(self):
            return conn_cm()

    with patch("src.database.connection.get_global_pool", return_value=DummyPool()):
        from src.database import connection as conn_mod

        res_all = conn_mod.fetch_all("select 1")
        assert res_all == [{"a": 1}]

        res_one = conn_mod.fetch_one("select 1")
        assert res_one == {"a": 1}

        # execute with commit=False should not call conn.commit
        conn_mod.execute("update x set y=1", commit=False)
        assert not fake_conn.commit.called


def test_bulk_upsert_technical_features_success():
    from src.database import db_utils

    rows = [
        {
            "ticker": "A",
            "date": "2020-01-01",
            "feature_category": "c",
            "feature_name": "n",
            "feature_value": 1,
        }
    ]

    fake_conn = MagicMock()
    fake_cur = MagicMock()
    fake_cur.rowcount = 1
    # psycopg2.execute_values expects cursor.connection.encoding to map to an
    # encoding; set a realistic attribute on the fake connection to avoid
    # psycopg2.extras.execute_values using MagicMock keys.
    fake_conn.cursor.return_value = fake_cur
    fake_conn.encoding = "UTF8"
    # also ensure cursor().connection.encoding is accessible
    fake_cur.connection = MagicMock()
    fake_cur.connection.encoding = "UTF8"

    @contextmanager
    def conn_cm():
        yield fake_conn

    class DummyPool:
        def connection(self):
            return conn_cm()

    def fake_execute_values(cur, sql, argslist, page_size=1000):
        # emulate psycopg2.extras.execute_values: set rowcount on cursor
        try:
            cur.rowcount = 1
        except Exception:
            pass

    with (
        patch("src.database.connection.get_global_pool", return_value=DummyPool()),
        patch(
            "src.database.connection.execute_values", side_effect=fake_execute_values
        ),
    ):
        res = db_utils.bulk_upsert_technical_features(rows)
        assert res == 1


def test_bulk_upsert_technical_features_rollback_on_exception():
    from src.database import db_utils

    rows = [
        {
            "ticker": "A",
            "date": "2020-01-01",
            "feature_category": "c",
            "feature_name": "n",
            "feature_value": 1,
        }
    ]

    fake_conn = MagicMock()
    fake_cur = MagicMock()
    # Make execute_values raise when psycopg2.extras.execute_values calls
    # the cursor; set cursor().connection.encoding to a real string and then
    # cause execute to raise.
    fake_cur.connection = MagicMock()
    fake_cur.connection.encoding = "UTF8"
    fake_cur.execute.side_effect = RuntimeError("boom")
    fake_conn.cursor.return_value = fake_cur

    @contextmanager
    def conn_cm():
        yield fake_conn

    class DummyPool:
        def connection(self):
            return conn_cm()

    def raising_execute_values(*args, **kwargs):
        raise RuntimeError("boom")

    with (
        patch("src.database.connection.get_global_pool", return_value=DummyPool()),
        patch(
            "src.database.db_utils.execute_values", side_effect=raising_execute_values
        ),
    ):
        with pytest.raises(RuntimeError):
            db_utils.bulk_upsert_technical_features(rows)
