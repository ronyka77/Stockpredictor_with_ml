
from src.data_collector.polygon_fundamentals import monitor as monitor_mod


def test_get_connection_pool_singleton(mocker):
    # Patch DatabaseConnectionPool to avoid real DB connections
    class DummyPool:
        def __init__(self, min_connections=2, max_connections=10):
            self.min = min_connections
            self.max = max_connections
            self.closed = False

        def close(self):
            self.closed = True

    mocker.patch('src.data_collector.polygon_fundamentals.db_pool.DatabaseConnectionPool', DummyPool)

    p1 = monitor_mod.get_connection_pool(2, 5)
    p2 = monitor_mod.get_connection_pool(2, 5)
    assert p1 is p2, 'get_connection_pool should return a singleton instance for same params'


def test_get_connection_pool_reinitializes_on_param_change(mocker):
    class DummyPool:
        def __init__(self, min_connections=2, max_connections=10):
            self.min = min_connections
            self.max = max_connections
            self.closed = False

        def close(self):
            self.closed = True

    mocker.patch('src.data_collector.polygon_fundamentals.db_pool.DatabaseConnectionPool', DummyPool)

    p1 = monitor_mod.get_connection_pool(2, 5)
    p2 = monitor_mod.get_connection_pool(3, 6)
    assert p1 is not p2, 'Connection pool should reinitialize when parameters change'


def test_fundamental_data_monitor_methods(mocker):
    # Create dummy pool with expected get_connection behaviour
    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, q, params=None):
            pass

        def fetchone(self):
            return {'count': 1} if 'COUNT' in '' else {'count': 1}

        def fetchall(self):
            return []

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class DummyPool:
        def get_connection(self):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield DummyConn()

            return _cm()

    mocker.patch('src.data_collector.polygon_fundamentals.monitor.get_connection_pool', return_value=DummyPool())
    mon = monitor_mod.FundamentalDataMonitor()
    prog = mon.get_collection_progress()
    assert 'total_tickers' in prog and 'tickers_with_data' in prog, 'Progress result should include counts'
    summary = mon.get_data_quality_summary()
    assert isinstance(summary, dict), 'Data quality summary should return a dict'
    recent = mon.get_recent_activity(days=7)
    assert isinstance(recent, list), 'Recent activity should return a list'


