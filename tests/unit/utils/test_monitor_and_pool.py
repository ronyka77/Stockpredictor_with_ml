from src.data_collector.polygon_fundamentals import monitor as monitor_mod


def test_fundamental_data_monitor_methods(mocker):
    """Exercise FundamentalDataMonitor methods and assert returned types/keys."""
    # Use canonical PoolFake via patch_global_pool
    from tests.fixtures.db import patch_global_pool, PoolFake

    pool = PoolFake()
    patch_global_pool(mocker, pool)
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
