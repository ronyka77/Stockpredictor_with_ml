"""
Database Connection Module

This module provides database connection functionality for the StockPredictor V1 system.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from typing import Iterable, Optional, Any, List, Tuple, Callable
from psycopg2.extras import execute_values as _execute_values
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
import os
import threading
import atexit
from threading import Semaphore

from src.utils.logger import get_logger

logger = get_logger(__name__, utility="database")


# --- Safe pooled manager + module-level lifecycle helpers ---
class ConnectionAcquireTimeout(RuntimeError):
    """Raised when acquiring a pooled connection times out."""


class PostgresConnection:
    """
    Thread-safe pooled connection wrapper that enforces an acquire timeout
    Uses a Semaphore to bound concurrent acquisition attempts and delegates
    actual connection management to psycopg2's ThreadedConnectionPool.
    """

    def __init__(self, minconn: int, maxconn: int, **conn_kwargs: Any):
        # Validate required credentials early
        if not conn_kwargs.get("password"):
            raise ValueError("DB_PASSWORD environment variable is required for database connections")

        self._minconn = minconn
        self._maxconn = maxconn
        # Ensure cursor_factory is set unless provided
        if "cursor_factory" not in conn_kwargs:
            conn_kwargs["cursor_factory"] = RealDictCursor
        self._pool = ThreadedConnectionPool(minconn, maxconn, **conn_kwargs)
        self._sem = Semaphore(maxconn)
        self._closed = False

        logger.info(
            f"Initialized PostgresConnection ({minconn}-{maxconn}) for "
            f"{conn_kwargs.get('host', '<dsn>')}:{conn_kwargs.get('port', '')}/{conn_kwargs.get('database', '')}"
        )

    @contextmanager
    def connection(
        self, timeout: float = 5.0
    ) -> Generator[psycopg2.extensions.connection, None, None]:
        """Acquire a connection from the pool with a timeout.

        Raises:
            ConnectionAcquireTimeout: if the semaphore cannot be acquired in time.
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        acquired = self._sem.acquire(timeout=timeout)
        if not acquired:
            logger.error("Timeout acquiring pooled connection")
            raise ConnectionAcquireTimeout("Timeout acquiring pooled connection")

        conn = None
        try:
            conn = self._pool.getconn()
            # Optional lightweight validation: ensure not closed
            if getattr(conn, "closed", 0):
                # return the broken connection and get a fresh one
                try:
                    self._pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = self._pool.getconn()

            yield conn
        except Exception as exc:
            # If an exception occurs while using the connection, roll back
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.error(f"Pooled connection error: {exc}")
            raise
        finally:
            if conn:
                try:
                    # Ensure no open transaction state is leaked
                    if hasattr(conn, "rollback") and callable(conn.rollback):
                        conn.rollback()
                except Exception:
                    pass
                try:
                    self._pool.putconn(conn)
                except Exception:
                    # If returning fails, attempt to close the raw connection
                    try:
                        conn.close()
                    except Exception:
                        pass
            # Release semaphore token regardless
            try:
                self._sem.release()
            except Exception:
                pass

    def close(self) -> None:
        """Close the underlying pool and prevent further acquisitions."""
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.closeall()
            logger.info("PostgresConnection closed")
        except Exception as exc:
            logger.error(f"Error closing pooled connections: {exc}")

    def get_connection(self, timeout: float = 5.0):
        """Compatibility wrapper for older `get_connection()` API.

        Returns the same contextmanager as `connection()`.
        """
        return self.connection(timeout=timeout)


# Module-level singleton management for a single pool instance
_GLOBAL_POOL: Optional[PostgresConnection] = None
# Use a reentrant lock to allow init routines to call each other without deadlocking
_GLOBAL_LOCK = threading.RLock()


def init_global_pool(minconn: int = 1, maxconn: int = 10) -> PostgresConnection:
    """Initialize and return the module-level global pool (idempotent).

    Always constructs connection parameters from environment variables. Call
    this at application startup. Returns the same instance on subsequent calls.
    """
    global _GLOBAL_POOL
    # Build connection kwargs from environment variables
    conn_kwargs = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "stock_data"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }

    if _GLOBAL_POOL is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_POOL is None:
                _GLOBAL_POOL = PostgresConnection(minconn, maxconn, **conn_kwargs)
    return _GLOBAL_POOL


def get_global_pool() -> PostgresConnection:
    """Return the initialized global pool, initializing it from env if needed."""
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        # Initialize with defaults if not already initialized
        with _GLOBAL_LOCK:
            if _GLOBAL_POOL is None:
                init_global_pool()
    return _GLOBAL_POOL


def close_global_pool() -> None:
    """Close and clear the global pool if present."""
    global _GLOBAL_POOL
    with _GLOBAL_LOCK:
        if _GLOBAL_POOL is not None:
            try:
                _GLOBAL_POOL.close()
            finally:
                _GLOBAL_POOL = None


# --- Convenience DB helpers ---

def fetch_all(
    query: str, params: Optional[Tuple] = None, dict_cursor: bool = True
) -> List[Any]:
    """Execute a read query and return all rows.

    Uses the global pool and a RealDictCursor by default.
    """
    pool = get_global_pool()
    with pool.connection() as conn:
        cur = (
            conn.cursor(cursor_factory=RealDictCursor) if dict_cursor else conn.cursor()
        )
        cur.execute(query, params or ())
        return cur.fetchall()


def fetch_one(
    query: str, params: Optional[Tuple] = None, dict_cursor: bool = True
) -> Optional[Any]:
    pool = get_global_pool()
    with pool.connection() as conn:
        cur = (
            conn.cursor(cursor_factory=RealDictCursor) if dict_cursor else conn.cursor()
        )
        cur.execute(query, params or ())
        return cur.fetchone()


def execute(query: str, params: Optional[Tuple] = None, commit: bool = True) -> None:
    pool = get_global_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
        if commit:
            conn.commit()


def execute_values(
    insert_sql: str, rows: Iterable[Tuple], page_size: int = 1000, commit: bool = True
) -> None:
    if not rows:
        return
    pool = get_global_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            _execute_values(cur, insert_sql, rows, page_size=page_size)
        if commit:
            conn.commit()


def run_in_transaction(fn: Callable[[Any, Any], Any]) -> Any:
    """Run a callable(fn) inside a transaction. fn(conn, cur) -> result

    Commits on success, rolls back and re-raises on error.
    """
    pool = get_global_pool()
    with pool.connection() as conn:
        cur = conn.cursor()
        try:
            res = fn(conn, cur)
            conn.commit()
            return res
        except Exception:
            conn.rollback()
            raise


# Ensure the pool is closed on normal interpreter exit
atexit.register(close_global_pool)
