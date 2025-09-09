import pytest
from unittest.mock import MagicMock
from contextlib import contextmanager

from src.database.connection import (
    PostgresConnection,
    ConnectionAcquireTimeout,
    run_in_transaction,
    get_global_pool, 
    close_global_pool
)


def test_databaseconnection_raises_when_password_missing():
    cfg = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": ""}
    with pytest.raises(ValueError) as exc:
        # pool init should validate missing password
        init_kwargs = cfg.copy()
        PostgresConnection(1, 1, **init_kwargs)
    assert "DB_PASSWORD" in str(exc.value), f"Expected missing password error; got: {exc.value}"


def test_safe_pooled_connection_timeout_and_release():
    fake_kwargs = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p"}
    sp = PostgresConnection(1, 1, **fake_kwargs)

    # Acquire internal semaphore to simulate exhaustion
    acquired = sp._sem.acquire(timeout=0)
    try:
        with pytest.raises(ConnectionAcquireTimeout):
            with sp.connection(timeout=0.01):
                pass
    finally:
        if acquired:
            sp._sem.release()


def test_get_connection_pool_singleton(mocker):
    # Use the shared PostgresPoolCompat fake from test helpers
    from tests._fixtures.helpers import PostgresPoolCompat

    mocker.patch(
        "src.database.connection.PostgresConnection",
        PostgresPoolCompat,
    )

    p1 = get_global_pool()
    p2 = get_global_pool()
    if p1 is not p2:
        raise AssertionError(
            "get_connection_pool should return a singleton instance for same params"
        )


def test_get_connection_pool_reinitializes_on_param_change(mocker):
    from tests._fixtures.helpers import PostgresPoolCompat

    mocker.patch(
        "src.database.connection.PostgresConnection",
        PostgresPoolCompat,
    )

    p1 = get_global_pool()
    # simulate reinit
    close_global_pool()
    p2 = get_global_pool()
    if p1 is p2:
        raise AssertionError(
            "Connection pool should reinitialize when parameters change"
        )


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

    mocker.patch("src.database.connection.get_global_pool", return_value=DummyPool())

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


