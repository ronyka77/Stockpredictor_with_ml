from src.data_collector.polygon_fundamentals import monitor as monitor_mod
from src.database.connection import get_global_pool, close_global_pool


def test_fundamental_data_monitor_methods(mocker):
    # Create dummy pool with expected get_connection behaviour
    # Reuse shared FakeDBPool / PostgresPoolCompat from centralized helpers
    from tests._fixtures.helpers import PostgresPoolCompat

    mocker.patch(
        "src.data_collector.polygon_fundamentals.monitor.get_global_pool",
        return_value=PostgresPoolCompat(),
    )
    mon = monitor_mod.FundamentalDataMonitor()
    prog = mon.get_collection_progress()
    if not ("total_tickers" in prog and "tickers_with_data" in prog):
        raise AssertionError("Progress result should include counts")
    summary = mon.get_data_quality_summary()
    if not isinstance(summary, dict):
        raise AssertionError("Data quality summary should return a dict")
    recent = mon.get_recent_activity(days=7)
    if not isinstance(recent, list):
        raise AssertionError("Recent activity should return a list")


def test_fundamental_data_monitor_methods(mocker):
    # Create dummy pool with expected get_connection behaviour
    from tests._fixtures.helpers import PostgresPoolCompat

    mocker.patch(
        "src.data_collector.polygon_fundamentals.monitor.get_global_pool",
        return_value=PostgresPoolCompat(),
    )
    mon = monitor_mod.FundamentalDataMonitor()
    prog = mon.get_collection_progress()
    if not ("total_tickers" in prog and "tickers_with_data" in prog):
        raise AssertionError("Progress result should include counts")
    summary = mon.get_data_quality_summary()
    if not isinstance(summary, dict):
        raise AssertionError("Data quality summary should return a dict")
    recent = mon.get_recent_activity(days=7)
    if not isinstance(recent, list):
        raise AssertionError("Recent activity should return a list")
