import pandas as pd


def make_sample_df():
    """Return a deterministic small DataFrame used by transformation tests."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ticker_id": [1, 1],
            "price": [100.0, 101.0],
            "volume": [1000, 1100],
        }
    )
    # enforce dtypes
    df["ticker_id"] = df["ticker_id"].astype(int)
    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df


def make_fake_http_response(status=200, json_data=None, raise_on_json=False):
    """Create a simple fake response object with .status_code and .json()"""
    class FakeResponse:
        def __init__(self, status, payload, raise_on_json):
            self.status_code = status
            self._payload = payload
            self._raise = raise_on_json

        def json(self):
            if self._raise:
                raise ValueError("malformed json")
            return self._payload

        def raise_for_status(self):
            if not (200 <= self.status_code < 300):
                raise Exception(f"HTTP {self.status_code}")

    return FakeResponse(status, json_data, raise_on_json)


class FakeLogicalStore:
    """A minimal logical store that records upserts for idempotency tests.

    This stores mapped rows produced by execute_values so tests can assert
    idempotency and final state without relying on a real DB.
    """

    def __init__(self):
        self._rows = {}

    def execute_values(self, query, values, template=None, page_size=100):
        # Accept either list of dict-like objects or list of tuples.
        if not values:
            return

        first = values[0]
        if isinstance(first, (tuple, list)):
            # attempt to parse column names from SQL query
            cols = []
            try:
                start = query.index("(") + 1
                end = query.index(")", start)
                cols = [c.strip() for c in query[start:end].split(",")]
            except Exception:
                # fallback: cannot map tuples to columns; store tuples under synthetic keys
                for v in values:
                    key = (v[0] if len(v) > 0 else None, v[1] if len(v) > 1 else None)
                    self._rows[key] = {f"col_{i}": val for i, val in enumerate(v)}
                return

            for v in values:
                mapped = {col: val for col, val in zip(cols, v)}
                key = (mapped.get("ticker"), mapped.get("date"))
                self._rows[key] = mapped
            return

        # assume dict-like
        for v in values:
            key = (v.get("ticker"), v.get("date"))
            self._rows[key] = v

    def unique_row_count(self):
        return len(self._rows)

    def fetch_all(self, query, *args, **kwargs):
        return list(self._rows.values())


class FakeConnection:
    """Combined fake connection and cursor used by connection-level tests.

    Provides cursor() that returns a context-manager cursor interface and tracks
    commits/rollbacks and executed statements.
    """

    def __init__(self):
        self.executed = []
        self._rowcount = 1
        self.commits = 0
        self.rollbacks = 0
        # compatibility: some tests access conn.cur
        self.cur = self

    def cursor(self, *args, **kwargs):
        # Return self as a simple cursor/context manager
        return self

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    @property
    def rowcount(self):
        return self._rowcount

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeConnectionPool:
    """Pool that returns FakeConnection instances via get_connection()."""

    def __init__(self):
        self.conn = FakeConnection()

    def get_connection(self):
        return self.conn
    
    def connection(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield self.conn

        return _cm()


class PostgresPoolCompat:
    """Compatibility wrapper that mimics minimal PostgresConnection used in tests.

    Provides a constructor signature that accepts min_connections/max_connections
    and exposes a `connection()` contextmanager and `close()` to integrate with
    code that expects the real PostgresConnection.
    """

    def __init__(self, min_connections=1, max_connections=10, **kwargs):
        self._pool = FakeConnectionPool()

    def close(self):
        # no-op for fake
        return None

    def connection(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield self._pool.conn

        return _cm()

    # compatibility alias used in some tests
    def get_connection(self):
        return self.connection()


class FakeThreadedPool:
    """Fake replacement for psycopg2.pool.ThreadedConnectionPool used in tests.

    Exposes getconn/putconn/closeall to match the real API and returns
    a FakeConnDB instance for compatibility with PostgresConnection.
    """

    def __init__(self, minconn=1, maxconn=10, **kwargs):
        self._conn = FakeConnection()

    def getconn(self):
        return self._conn

    def putconn(self, conn, close=False):
        # no-op for the fake
        return None

    def closeall(self):
        return None


def make_fake_db(mode: str = "logical"):
    """Factory to create a fake DB helper.

    mode options:
      - "logical": returns FakePool (logical upsert store)
      - "connection": returns FakeDBPool (connection pool with FakeConnDB)
      - "threaded": returns FakeThreadedPool (psycopg2 ThreadedConnectionPool-like)
      - "postgres": returns PostgresPoolCompat (high-level PostgresConnection shim)
    """
    mode = mode.lower()
    if mode == "logical":
        return FakeLogicalStore()
    if mode == "connection":
        return FakeConnectionPool()
    if mode == "threaded":
        return FakeThreadedPool()
    if mode == "postgres":
        return PostgresPoolCompat()
    raise ValueError(f"Unknown fake DB mode: {mode}")


# Backwards-compatible aliases for older tests
FakePool = FakeLogicalStore
FakeDBPool = FakeConnectionPool
PoolCompat = PostgresPoolCompat


