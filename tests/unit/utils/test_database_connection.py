import pytest

from src.database.connection import DatabaseConnection, DatabaseConnectionPool


def test_database_connection_init_requires_password():
    with pytest.raises(ValueError, match="DB_PASSWORD"):
        DatabaseConnection(config={
            'host': 'h', 'port': 5432, 'database': 'db', 'user': 'u', 'password': ''
        })


def test_get_connection_string_excludes_password():
    cfg = {'host': 'dbhost', 'port': 5432, 'database': 'stock', 'user': 'u', 'password': 'secret'}
    db = DatabaseConnection(config=cfg)
    s = db.get_connection_string()
    assert 'secret' not in s and s == 'dbhost:5432/stock', f"Unexpected connection string: {s}"


def test_test_connection_returns_false_on_exception(mocker):
    cfg = {'host': 'h', 'port': 5432, 'database': 'db', 'user': 'u', 'password': 'p'}
    mocker.patch('src.database.connection.psycopg2.connect', side_effect=Exception('conn failed'))
    db = DatabaseConnection(config=cfg)
    assert db.test_connection() is False, 'test_connection should return False on connection exception'


def test_test_connection_returns_true_on_success(mocker):
    cfg = {'host': 'h', 'port': 5432, 'database': 'db', 'user': 'u', 'password': 'p'}

    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, q):
            pass

        def fetchone(self):
            return (1,)


    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def close(self):
            pass

    mocker.patch('src.database.connection.psycopg2.connect', return_value=DummyConn())
    db = DatabaseConnection(config=cfg)
    assert db.test_connection() is True, 'test_connection should return True when SELECT 1 succeeds'


def test_pool_init_requires_password():
    with pytest.raises(ValueError, match="DB_PASSWORD"):
        DatabaseConnectionPool(config={
            'host': 'h', 'port': 5432, 'database': 'db', 'user': 'u', 'password': ''
        })


