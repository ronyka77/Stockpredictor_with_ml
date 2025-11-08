import pytest

from src.data_collector.polygon_fundamentals_v2.repository import FundamentalsRepository


class FakeCursor:
    def __init__(self):
        self.executed = []
        self._rowcount = 1

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


class FakeConn:
    def __init__(self):
        self.cur = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def pool_fake():
    from tests.fixtures.db import PoolFake

    return PoolFake()


@pytest.mark.unit
def test_upsert_raw_payload_idempotent(mocker, pool_fake):
    """Upsert raw payload twice should be idempotent and commit twice"""
    repo = FundamentalsRepository()
    mocker.patch.object(repo, "pool", pool_fake)
    # Also ensure module-level helpers use the same fake pool
    from tests.fixtures import patch_global_pool

    patch_global_pool(mocker, repo.pool)

    payload = {"results": [{"x": 1}], "status": "OK"}
    repo.upsert_raw_payload(
        ticker_id=1,
        period_end="2024-01-01",
        timeframe="quarter",
        fiscal_period="Q1",
        fiscal_year="2024",
        filing_date="2024-02-15",
        source="api",
        payload=payload,
    )

    # Running again with same payload should hit ON CONFLICT and not error
    repo.upsert_raw_payload(
        ticker_id=1,
        period_end="2024-01-01",
        timeframe="quarter",
        fiscal_period="Q1",
        fiscal_year="2024",
        filing_date="2024-02-15",
        source="api",
        payload=payload,
    )

    cur = repo.pool.conn.cur
    # Ensure we executed twice and committed twice
    if repo.pool.conn.commits != 2:
        raise AssertionError("Expected two commits after two upserts")
    if len(cur.executed) != 2:
        raise AssertionError("Expected two executed statements in cursor")
    # Ensure parameters include JSON string for payload
    _, params = cur.executed[-1]
    if not isinstance(params["payload"], str):
        raise AssertionError("Payload parameter should be a JSON string")
