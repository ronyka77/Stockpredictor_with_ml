"""Canonical DB test fakes: PoolFake and ConnectionFake.

These provide a single in-memory backing store and two primary test doubles:
- ConnectionFake: exposes cursor(), commit/rollback, execute_values, and a
    simple cursor API used by higher-level helpers.
- PoolFake: exposes connection() contextmanager, getconn/putconn, closeall

Compatibility aliases are provided at the bottom so older test imports can
continue to work during incremental migration.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple


class CursorFake:
    def __init__(self, conn: "ConnectionFake"):
        self._conn = conn
        self._last_query = None

    def execute(self, sql: str, params: Optional[Tuple] = None):
        self._last_query = (sql, params)
        # Record executed statement for inspection
        self._conn.executed.append((sql, params))

    def fetchone(self):
        # Return first row from logical store values if present
        rows = list(self._conn._store.values())
        return rows[0] if rows else None

    def fetchall(self):
        return list(self._conn._store.values())

    @property
    def rowcount(self):
        return len(self._conn._store)

    # Support context manager protocol for `with conn.cursor() as cur:` usage
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # No special handling; allow exception propagation by returning False
        return False


class ConnectionFake:
    """Connection-like fake with in-memory backing store and cursor support."""

    def __init__(self):
        self.executed: List[Tuple[Any, Any]] = []
        self._store: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        self.commits = 0
        self.rollbacks = 0
        # compatibility alias used in some tests
        self.cur = self

    def cursor(self, *args, **kwargs):
        return CursorFake(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        # no-op for fake
        return None

    # Convenience: support execute_values-based upsert used in many tests
    def execute_values(
        self,
        query: str,
        rows: Iterable
    ):
        # Accept dict-like values or tuples. Attempt to map to columns if tuples.
        rows = list(rows)
        if not rows:
            return

        first = rows[0]
        if isinstance(first, (list, tuple)):
            cols = self._parse_columns_from_query(query)
            if cols is None:
                self._store_tuple_fallback(rows)
                return

            self._store_tuple_rows(rows, cols)
            return

        # assume dict-like
        for v in rows:
            key = (v.get("ticker"), v.get("date"))
            self._store[key] = v

    def _parse_columns_from_query(self, query: str) -> Optional[List[str]]:
        """Attempt to parse column names from an INSERT query string.

        Returns a list of column names or None if parsing fails.
        """
        try:
            start = query.index("(") + 1
            end = query.index(")", start)
            cols = [c.strip() for c in query[start:end].split(",")]
            return cols
        except Exception:
            return None

    def _store_tuple_rows(self, rows: List[tuple], cols: List[str]) -> None:
        """Store tuple rows by zipping with parsed column names."""
        for v in rows:
            mapped = dict(zip(cols, v))
            key = (mapped.get("ticker"), mapped.get("date"))
            self._store[key] = mapped

    def _store_tuple_fallback(self, rows: List[tuple]) -> None:
        """Fallback storage when column parsing fails: store tuples under synthetic keys."""
        for v in rows:
            key = (v[0] if len(v) > 0 else None, v[1] if len(v) > 1 else None)
            self._store[key] = {f"col_{i}": val for i, val in enumerate(v)}


class PoolFake:
    """Pool fake that returns ConnectionFake instances via a contextmanager.

    Implements getconn/putconn/closeall for compatibility with threaded pools.
    """

    def __init__(self, *args, **kwargs):
        # Accept compatibility constructor args (minconn, maxconn, **conn_kwargs)
        # but otherwise ignore them; keep list of created connections for inspection
        self._conns: List[ConnectionFake] = []
        # Compatibility: expose last active connection as `conn` to match some tests
        self.conn: Optional[ConnectionFake] = None

    @contextmanager
    def connection(self, timeout: Optional[float] = None):
        # Reuse a single connection instance for connection() to mimic a simple
        # compatibility pool used in many unit tests where they inspect
        # repo.pool.conn for executed statements/commits across multiple calls.
        if not self._conns:
            conn = ConnectionFake()
            self._conns.append(conn)
        else:
            conn = self._conns[0]
        # expose last (or single) connection for tests that inspect repo.pool.conn
        self.conn = conn
        try:
            yield conn
        finally:
            # implicit rollback to avoid leaking transaction state in tests
            try:
                conn.rollback()
            except Exception:
                # ignore rollback failures in tests
                pass

    def getconn(self):
        # return a new connection compatible with getconn()
        conn = ConnectionFake()
        self._conns.append(conn)
        # expose last connection for compatibility
        self.conn = conn
        return conn

    def putconn(self, conn, close: bool = False):
        # no-op in fake
        try:
            if close:
                conn.close()
        except Exception:
            pass

    def closeall(self):
        # no-op for fake
        return None

    def execute_values(
        self,
        insert_sql: str,
        rows: Iterable[Tuple],
    ):
        # Convenience proxy: use a transient connection to perform execute_values
        conn = ConnectionFake()
        conn.execute_values(insert_sql, rows)
        return None


def patch_global_pool(mocker, pool):
    """Patch module-level pool helpers to return the provided `pool`.

    Mirrors older test helper behavior but uses the canonical PoolFake.
    """

    def init_global_pool_fake(minconn: int = 1, maxconn: int = 10):
        return pool

    def get_global_pool_fake():
        return pool

    def close_global_pool_fake():
        return None

    mocker.patch("src.database.connection.init_global_pool", init_global_pool_fake)
    mocker.patch("src.database.connection.get_global_pool", get_global_pool_fake)
    mocker.patch("src.database.connection.close_global_pool", close_global_pool_fake)
