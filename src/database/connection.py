"""
Database Connection Module

This module provides database connection functionality for the StockPredictor V1 system.
"""

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from typing import Iterable, Optional, Any, List, Tuple, Callable, Union, Dict
from contextlib import contextmanager
from typing import Generator
import os
import threading
import atexit
from threading import Semaphore
import itertools
import time
from dataclasses import dataclass

from src.utils.core.logger import get_logger

logger = get_logger(__name__, utility="database")


@dataclass
class PoolStats:
    """Connection pool statistics for monitoring."""
    min_size: int
    max_size: int
    current_size: int
    available_connections: int
    used_connections: int
    waiting_requests: int
    total_connections_created: int
    total_connections_destroyed: int
    last_health_check: float
    health_check_success: bool


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
        self._last_health_check = 0.0
        self._health_check_success = True
        self._health_check_interval = 30.0  # Check health every 30 seconds

    def _perform_health_check(self) -> bool:
        """Perform a health check on the database connection pool.

        Returns:
            bool: True if health check passes, False otherwise.
        """
        try:
            with self._pool.connection() as conn:
                # Simple query to test connection health
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 as health_check")
                    result = cur.fetchone()
                    return result is not None and result[0] == 1
        except Exception as exc:
            logger.warning(f"Database health check failed: {exc}")
            return False

    def check_health(self, force: bool = False) -> bool:
        """Check database connection health, with optional forced recheck.

        Args:
            force: If True, perform health check regardless of timing.

        Returns:
            bool: True if database is healthy, False otherwise.
        """
        current_time = time.time()

        # Skip health check if recently performed and not forced
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            return self._health_check_success

        self._health_check_success = self._perform_health_check()
        self._last_health_check = current_time

        if not self._health_check_success:
            logger.error("Database health check failed - connection pool may need recovery")
        else:
            logger.debug("Database health check passed")

        return self._health_check_success

    def validate_connection(self, conn: Any) -> bool:
        """Validate a connection before use and attempt recovery if invalid.

        Args:
            conn: Database connection to validate.

        Returns:
            bool: True if connection is valid, False otherwise.
        """
        # Basic closed check
        if getattr(conn, "closed", False):
            logger.debug("Connection is closed, marking as invalid")
            return False

        # Perform health check if needed
        if not self.check_health():
            logger.warning("Database health check failed during connection validation")
            return False

        # Try a simple query to validate connection
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result is not None and result[0] == 1
        except Exception as exc:
            logger.warning(f"Connection validation failed: {exc}")
            return False

    def get_pool_stats(self) -> PoolStats:
        """Get current connection pool statistics.

        Returns:
            PoolStats: Current pool statistics.
        """
        # Get basic pool information
        try:
            # Try to get stats from psycopg_pool if available
            if hasattr(self._pool, 'get_stats'):
                stats = self._pool.get_stats()
                current_size = getattr(stats, 'size', self._maxconn)
                available = getattr(stats, 'available', 0)
                used = current_size - available
                waiting = getattr(stats, 'waiting', 0)
                created = getattr(stats, 'connections_created', 0)
                destroyed = getattr(stats, 'connections_destroyed', 0)
            else:
                # Fallback estimates
                current_size = self._maxconn  # Conservative estimate
                available = self._sem._value if hasattr(self._sem, '_value') else 0
                used = max(0, current_size - available)
                waiting = max(0, self._maxconn - available)
                created = 0  # Not available
                destroyed = 0  # Not available
        except Exception:
            # If stats collection fails, provide basic info
            current_size = self._maxconn
            available = 0
            used = 0
            waiting = 0
            created = 0
            destroyed = 0

        return PoolStats(
            min_size=self._minconn,
            max_size=self._maxconn,
            current_size=current_size,
            available_connections=available,
            used_connections=used,
            waiting_requests=waiting,
            total_connections_created=created,
            total_connections_destroyed=destroyed,
            last_health_check=self._last_health_check,
            health_check_success=self._health_check_success
        )

    @contextmanager
    def connection(self, timeout: float = 5.0) -> Generator[Any, None, None]:
        """Acquire a connection from the pool with a timeout.

        Raises:
            RuntimeError: if the semaphore cannot be acquired in time.
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        # Use the pool's context manager to get a connection when possible,
        # but preserve backward compatibility with pools that expose getconn/
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
                # Enhanced validation with health monitoring and recovery
                if not self.validate_connection(conn):
                    logger.debug("Invalid connection detected, attempting recovery")
                    # Return the broken connection and get a fresh one
                    try:
                        self._pool.putconn(conn, close=True)
                    except Exception:
                        pass
                    conn = self._pool.getconn()
                    # Validate the new connection
                    if not self.validate_connection(conn):
                        raise RuntimeError("Failed to obtain valid connection after recovery attempt")

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
                        # Enhanced validation with health monitoring
                        if not self.validate_connection(conn):
                            raise RuntimeError("Acquired invalid connection from pool - health check failed")
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


def check_database_health() -> bool:
    """Check the health of the global database connection pool.

    Returns:
        bool: True if database is healthy, False otherwise.
    """
    try:
        pool = get_global_pool()
        return pool.check_health(force=True)
    except Exception as exc:
        logger.error(f"Database health check failed: {exc}")
        return False


def get_database_stats() -> PoolStats:
    """Get statistics for the global database connection pool.

    Returns:
        PoolStats: Current pool statistics.
    """
    try:
        pool = get_global_pool()
        return pool.get_pool_stats()
    except Exception as exc:
        logger.error(f"Failed to get database stats: {exc}")
        # Return basic fallback stats
        return PoolStats(
            min_size=1,
            max_size=10,
            current_size=0,
            available_connections=0,
            used_connections=0,
            waiting_requests=0,
            total_connections_created=0,
            total_connections_destroyed=0,
            last_health_check=time.time(),
            health_check_success=False
        )


def fetch_all(query: str, params: Optional[Tuple] = None, dict_cursor: bool = True) -> List[Any]:
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


def execute(query: str, params: Optional[Union[Tuple, Dict[str, Any]]] = None, commit: bool = True) -> None:
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
