"""
Database Connection Module

This module provides database connection functionality for the StockPredictor V1 system.
"""

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from typing import Iterable, Optional, Any, List, Tuple, Callable
from contextlib import contextmanager
from typing import Generator
import os
import threading
import atexit
from threading import Semaphore
import itertools

from src.utils.logger import get_logger

logger = get_logger(__name__, utility="database")


class PostgresConnection:
    """Thread-safe pooled connection wrapper that enforces an acquire timeout.

    Uses a Semaphore to bound concurrent acquisition attempts and delegates
    actual connection management to psycopg3's `ConnectionPool`.
    """

    def __init__(self, minconn: int, maxconn: int, **conn_kwargs: Any):
        # Validate required credentials early
        if not conn_kwargs.get("password"):
            raise ValueError(
                "DB_PASSWORD environment variable is required for database connections"
            )

        self._minconn = minconn
        self._maxconn = maxconn

        dsn_parts: List[str] = []
        for k, v in conn_kwargs.items():
            if v is None or v == "":
                continue
            val = str(v)
            if " " in val:
                val = f"'{val}'"
            dsn_parts.append(f"{k}={val}")
        dsn = " ".join(dsn_parts)
        # Use the alias (ThreadedConnectionPool) so unit tests that patch that
        # symbol receive the fake pool instead of creating a real ConnectionPool.
        self._pool = ThreadedConnectionPool(conninfo=dsn, min_size=minconn, max_size=maxconn)
        self._sem = Semaphore(maxconn)
        self._closed = False

    @contextmanager
    def connection(
        self, timeout: float = 5.0
    ) -> Generator[Any, None, None]:
        """Acquire a connection from the pool with a timeout.

        Raises:
            RuntimeError: if the semaphore cannot be acquired in time.
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        # Use the pool's context manager to get a connection when possible,
        # but preserve backward compatibility with pools that expose getconn/
        # putconn (psycopg2 ThreadedConnectionPool) for tests and older callers.
        acquired = self._sem.acquire(timeout=timeout)
        if not acquired:
            logger.error("Timeout acquiring pooled connection")
            raise RuntimeError("Timeout acquiring pooled connection")

        # If the underlying pool exposes getconn/putconn, use the legacy flow
        # so tests that inject fake pools (MagicMock with getconn/putconn) work.
        if hasattr(self._pool, "getconn") and hasattr(self._pool, "putconn"):
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
        else:
            # New-style pool with context manager (psycopg_pool.ConnectionPool)
            try:
                with self._pool.connection() as conn:
                    try:
                        if getattr(conn, "closed", False):
                            raise RuntimeError("Acquired closed connection from pool")
                        yield conn
                    except Exception:
                        try:
                            if hasattr(conn, "rollback") and callable(conn.rollback):
                                conn.rollback()
                        except Exception:
                            pass
                        logger.error("Pooled connection error")
                        raise
            finally:
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
            # psycopg_pool exposes a `close()` method to shut down the pool
            # and release resources.
            try:
                self._pool.close()
            except Exception:
                # Fallback: some pool versions provide `closeall()`
                try:
                    self._pool.closeall()
                except Exception:
                    pass
            # logger.info("PostgresConnection closed")
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
        "dbname": os.getenv("DB_NAME", "stock_data"),
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
        # Use psycopg3 row factory to return mapping objects when requested.
        cur = conn.cursor(row_factory=dict_row) if dict_cursor else conn.cursor()
        cur.execute(query, params or ())
        return cur.fetchall()


def fetch_one(
    query: str, params: Optional[Tuple] = None, dict_cursor: bool = True
) -> Optional[Any]:
    pool = get_global_pool()
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row) if dict_cursor else conn.cursor()
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
    """Efficiently insert many rows in pages.

    The `insert_sql` argument is expected to contain a single ``%s`` placeholder
    where the expanded VALUES list will be inserted. Rows are passed as an
    iterable of tuples and executed with parameter binding for safety.

    Example:
        insert_sql = "INSERT INTO t (a,b) VALUES %s ON CONFLICT ..."
    """
    rows = list(rows)
    if not rows:
        return
    pool = get_global_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Determine number of columns from the first row and build a
            # placeholder template like "(%s,%s,...)" for each row.
            num_cols = len(rows[0])
            value_template = "(" + ",".join(["%s"] * num_cols) + ")"
            for i in range(0, len(rows), page_size):
                page = rows[i : i + page_size]
                # Create a VALUES section with the correct number of row
                # placeholders and flatten the parameters into a single tuple.
                values_placeholders = ",".join([value_template] * len(page))
                sql = insert_sql % values_placeholders
                params = tuple(itertools.chain.from_iterable(page))
                cur.execute(sql, params)
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

# Expose the old symbol name so external patches continue to work.
ThreadedConnectionPool = ConnectionPool

# Ensure the pool is closed on normal interpreter exit
atexit.register(close_global_pool)


