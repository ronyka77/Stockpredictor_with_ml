import json

import pytest
from tests.unit.data_collector.polygon_news.helpers import (
    load_fixture,
    processed_article_expected as _processed_article_expected,
)


@pytest.fixture(scope="function")
def db_session():
    """Compatibility fixture retained for tests that request `db_session`.

    After migration away from SQLAlchemy, tests should avoid using this fixture.
    It provides a minimal session-like object with commit/rollback/close no-ops
    so older tests that still request it won't fail during collection.
    """

    class DummySession:
        def commit(self):
            return None

        def rollback(self):
            return None

        def close(self):
            return None

        # Minimal query() placeholder to raise if used; tests should not rely on ORM
        def query(self, *a, **k):
            raise RuntimeError("ORM query not supported in pool-based tests")

    yield DummySession()


def _load_fixture(name: str):
    path = f"tests/_fixtures/data/{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        # ensure published_utc strings are returned as ISO strings (validator will parse)
        return data


@pytest.fixture
def sample_raw_article_full():
    return load_fixture("sample_article_full")


@pytest.fixture
def sample_raw_article_missing():
    return load_fixture("sample_article_missing_fields")


@pytest.fixture
def sample_raw_article_malformed():
    return load_fixture("sample_article_malformed_types")


@pytest.fixture
def processed_article_expected():
    # Expected processed result for sample_article_full
    return _processed_article_expected()
