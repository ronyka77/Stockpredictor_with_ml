import pandas as pd
import pytest
from types import SimpleNamespace
from datetime import datetime, date, timedelta

from src.data_collector.indicator_pipeline.indicator_pipeline import (
    BatchFeatureProcessor,
    BatchJobConfig,
    _process_ticker_worker,
)


@pytest.fixture
def processor():
    p = BatchFeatureProcessor()
    yield p
    p.close()


@pytest.fixture
def sample_stock_df():
    # 10 days of simple OHLCV data
    idx = pd.date_range(end=datetime.today(), periods=10, freq="D")
    df = pd.DataFrame(
        {
            "open": range(10),
            "high": range(1, 11),
            "low": range(0, 10),
            "close": range(1, 11),
            "volume": [100] * 10,
        },
        index=idx,
    )
    return df


def test_get_available_tickers_calls_loader(mocker, processor):
    """Call data loader to obtain available tickers"""
    mocker.patch.object(
        processor.data_loader, "get_available_tickers", return_value=["A", "B"]
    )
    tickers = processor.get_available_tickers(min_data_points=1)
    assert tickers == ["A", "B"]


def test_process_single_ticker_no_data(mocker, processor):
    """Return error when loader provides no data for a ticker"""
    # Make loader return empty DataFrame
    mocker.patch.object(
        processor.data_loader, "load_stock_data", return_value=pd.DataFrame()
    )
    cfg = BatchJobConfig(
        start_date="2025-01-01", end_date="2025-01-02", min_data_points=1
    )
    res = processor.process_single_ticker("T", cfg, job_id="j1")
    assert res["success"] is False
    assert res["error"] == "No data available"


def make_feature_result():
    # features DataFrame with two feature columns; include a NaN, inf, and huge value to test filtering
    idx = pd.date_range(end=datetime.today(), periods=3, freq="D")
    feats = pd.DataFrame(
        {
            "feat_a": [1.0, float("nan"), 2.0],
            "feat_b": [float("inf"), 3.0, 1e12],
        },
        index=idx,
    )
    return SimpleNamespace(data=feats, warnings=["w1"], quality_score=0.95)


def test_process_single_ticker_saves_to_parquet_and_db(
    processor, sample_stock_df, mocker, caplog
):
    """Calculate features, save to parquet and perform DB bulk upsert"""
    # Arrange
    mocker.patch.object(
        processor.data_loader, "load_stock_data", return_value=sample_stock_df
    )
    mock_feature_result = make_feature_result()
    mocker.patch.object(
        processor.feature_calculator,
        "calculate_all_features",
        return_value=mock_feature_result,
    )

    # Patch storage save and DB bulk upsert (patch upstream db util)
    save_mock = mocker.patch.object(processor.feature_storage, "save_features")
    bulk_upsert_mock = mocker.patch(
        "src.database.db_utils.bulk_upsert_technical_features", return_value=5
    )

    cfg = BatchJobConfig(
        start_date=(date.today() - timedelta(days=10)),
        end_date=date.today(),
        min_data_points=1,
        save_to_parquet=True,
        save_to_database=True,
        overwrite_existing=True,
    )

    # Act
    caplog.set_level("INFO")
    res = processor.process_single_ticker("TICK", cfg, job_id="job1")

    # Assert
    assert res["success"] is True
    assert res["features_calculated"] == mock_feature_result.data.shape[1]
    save_mock.assert_called_once()
    assert bulk_upsert_mock.called
    args, kwargs = bulk_upsert_mock.call_args
    rows_passed = args[0]
    assert all(
        not (
            pd.isna(r["feature_value"])
            or r["feature_value"] == float("inf")
            or abs(r["feature_value"]) >= 1e9
        )
        for r in rows_passed
    )


def test_batchjobconfig_parses_dates():
    """Parse start and end dates from ISO strings and datetimes"""
    # ISO string
    cfg1 = BatchJobConfig(start_date="2025-01-01", end_date="2025-01-02")
    assert isinstance(cfg1.start_date, date) and isinstance(cfg1.end_date, date)

    # datetime objects
    cfg2 = BatchJobConfig(
        start_date=datetime(2025, 1, 1), end_date=datetime(2025, 1, 2)
    )
    assert isinstance(cfg2.start_date, date) and isinstance(cfg2.end_date, date)


def test_process_batch_aggregates_results(processor, mocker):
    """Aggregate batch processing results with successes and failures"""
    # Arrange: patch process_single_ticker to simulate one success and one failure
    success = {"ticker": "A", "success": True, "features_calculated": 2, "warnings": 0}
    failure = {"ticker": "B", "success": False, "error": "bad"}
    mocker.patch.object(
        processor, "process_single_ticker", side_effect=[success, failure]
    )
    cfg = BatchJobConfig(
        batch_size=2, max_workers=2, use_processes=False, min_data_points=1
    )

    # Act
    summary = processor.process_batch(["A", "B"], cfg)

    # Assert
    assert summary["total_tickers"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1
    assert summary["success_rate"] == pytest.approx((1 / 2) * 100)


def test__process_ticker_worker_no_data(mocker):
    """Worker returns error when no data is loaded for the ticker"""
    # Arrange: patch loader.load_stock_data in the worker's module path
    mocker.patch(
        "src.data_collector.indicator_pipeline.indicator_pipeline.StockDataLoader.load_stock_data",
        return_value=pd.DataFrame(),
    )
    config_dict = {
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "min_data_points": 1,
        "save_to_parquet": False,
        "save_to_database": False,
    }
    res = _process_ticker_worker("T", config_dict, job_id="j")
    assert res["success"] is False
    assert res["error"] == "No data available"


def test__process_ticker_worker_insufficient_data(mocker):
    """Worker errors on insufficient historical data for calculations"""
    # patch the StockDataLoader used by the worker to return a small dataframe
    df = pd.DataFrame({"a": [1]})
    mock_loader_cls = mocker.Mock()
    mock_loader = mock_loader_cls.return_value
    mock_loader.load_stock_data.return_value = df

    # Patch the module where the worker imports StockDataLoader
    mocker.patch("src.feature_engineering.data_loader.StockDataLoader", mock_loader_cls)

    config_dict = {
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "min_data_points": 10,
        "save_to_parquet": False,
        "save_to_database": False,
    }
    res = _process_ticker_worker("T", config_dict, job_id="j")
    assert res["success"] is False
    assert "Insufficient data" in res["error"]


def test__process_ticker_worker_success_saves_db(mocker):
    """Worker saves calculated features and performs DB upsert on success"""
    # Prepare loader, calculator, and storage mocks used when worker imports inside function
    idx = pd.date_range(end=datetime.today(), periods=3, freq="D")
    feats = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=idx)
    mock_calc = mocker.Mock()
    mock_calc.calculate_all_features.return_value = SimpleNamespace(
        data=feats, warnings=[], quality_score=0.9
    )

    mock_loader_cls = mocker.Mock()
    mock_loader = mock_loader_cls.return_value
    mock_loader.load_stock_data.return_value = pd.DataFrame({"open": [1, 2, 3]})

    mock_storage_cls = mocker.Mock()
    mock_storage = mock_storage_cls.return_value
    mock_storage.save_features = mocker.Mock()

    # Patch the imported classes/functions inside the worker
    mocker.patch("src.feature_engineering.data_loader.StockDataLoader", mock_loader_cls)
    mocker.patch(
        "src.data_collector.indicator_pipeline.feature_calculator.FeatureCalculator",
        return_value=mock_calc,
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.feature_storage.FeatureStorage",
        mock_storage_cls,
    )

    # Patch DB bulk upsert
    bulk_upsert_mock = mocker.patch(
        "src.database.db_utils.bulk_upsert_technical_features", return_value=10
    )

    config_dict = {
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "min_data_points": 1,
        "save_to_parquet": False,
        "save_to_database": True,
        "feature_categories": None,
        "overwrite_existing": True,
    }
    res = _process_ticker_worker("TICK", config_dict, job_id="j1")
    assert res["success"] is True
    assert res["features_calculated"] == feats.shape[1]
    assert bulk_upsert_mock.called


def test__save_features_to_database_filters_and_returns_count(mocker, processor):
    """Filter invalid feature values before bulk upsert and return count"""
    # Prepare a feature_result with NaN, inf and huge values
    idx = pd.date_range(end=datetime.today(), periods=3, freq="D")
    feats = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0],
            "nan_col": [1.0, float("nan"), 2.0],
            "inf_col": [1.0, float("inf"), 2.0],
            "huge": [1.0, 2.0, 1e12],
        },
        index=idx,
    )

    feature_result = SimpleNamespace(data=feats, warnings=[], quality_score=0.5)

    bulk_upsert_mock = mocker.patch(
        "src.database.db_utils.bulk_upsert_technical_features", return_value=7
    )

    saved = processor._save_features_to_database(
        "TICK", feature_result, job_id="j", overwrite=True
    )
    assert saved == 7
    # ensure rows passed do not contain NaN or inf or huge values
    args, _ = bulk_upsert_mock.call_args
    rows = args[0]
    assert all(
        not (
            pd.isna(r["feature_value"])
            or r["feature_value"] == float("inf")
            or abs(r["feature_value"]) >= 1e9
        )
        for r in rows
    )


def test_process_single_ticker_insufficient_data(processor, mocker):
    """Process single ticker reports insufficient data when too few rows"""
    # loader returns small DF
    small_df = pd.DataFrame({"a": [1]})
    mocker.patch.object(processor.data_loader, "load_stock_data", return_value=small_df)
    cfg = BatchJobConfig(
        start_date="2025-01-01", end_date="2025-01-02", min_data_points=5
    )
    res = processor.process_single_ticker("T", cfg, job_id="j1")
    assert res["success"] is False
    assert "Insufficient data" in res["error"]


def test_process_single_ticker_db_error_logs_but_succeeds(
    processor, sample_stock_df, mocker
):
    """Log DB errors during upsert but still succeed overall"""
    mocker.patch.object(
        processor.data_loader, "load_stock_data", return_value=sample_stock_df
    )
    mock_feature_result = make_feature_result()
    mocker.patch.object(
        processor.feature_calculator,
        "calculate_all_features",
        return_value=mock_feature_result,
    )

    # make bulk upsert raise
    mocker.patch(
        "src.database.db_utils.bulk_upsert_technical_features",
        side_effect=RuntimeError("db down"),
    )
    # patch logger.error to observe it's called
    log_mock = mocker.patch(
        "src.data_collector.indicator_pipeline.indicator_pipeline.logger"
    )

    cfg = BatchJobConfig(
        start_date=(date.today() - timedelta(days=10)),
        end_date=date.today(),
        min_data_points=1,
        save_to_parquet=False,
        save_to_database=True,
        overwrite_existing=True,
    )
    res = processor.process_single_ticker("TICK", cfg, job_id="job1")
    assert res["success"] is True
    # logger.error should have been called at least once
    assert log_mock.error.call_count >= 1


def test_run_production_batch_consolidation_paths(mocker):
    """Run production batch and follow consolidation success path"""
    # Patch processor.get_available_tickers to return some tickers
    module = __import__(
        "src.data_collector.indicator_pipeline.indicator_pipeline",
        fromlist=["BatchFeatureProcessor"],
    )
    processor_cls = getattr(module, "BatchFeatureProcessor")
    proc_inst = processor_cls()
    mocker.patch.object(proc_inst, "get_available_tickers", return_value=["A", "B"])

    # Patch FeatureStorage.remove_all_versions_for_all_tickers
    module_fs = __import__(
        "src.data_collector.indicator_pipeline.indicator_pipeline",
        fromlist=["FeatureStorage"],
    )
    storage_cls = getattr(module_fs, "FeatureStorage")
    mocker.patch.object(storage_cls, "remove_all_versions_for_all_tickers")

    # Patch processor.process_batch to return result with successful>0 and success_rate
    mocker.patch.object(
        proc_inst,
        "process_batch",
        return_value={
            "total_tickers": 2,
            "successful": 2,
            "failed": 0,
            "results": [
                {"ticker": "A", "success": True},
                {"ticker": "B", "success": True},
            ],
            "total_features": 4,
            "success_rate": 100.0,
        },
    )

    # Patch consolidate_existing_features success (include compression_ratio)
    mocker.patch(
        "src.data_collector.indicator_pipeline.consolidated_storage.consolidate_existing_features",
        return_value={
            "files_created": 1,
            "total_size_mb": 1.0,
            "compression_ratio": 1.0,
            "files": [],
        },
    )

    # Patch BatchFeatureProcessor to return our proc_inst when constructed
    mocker.patch(
        "src.data_collector.indicator_pipeline.indicator_pipeline.BatchFeatureProcessor",
        return_value=proc_inst,
    )

    from src.data_collector.indicator_pipeline.indicator_pipeline import (
        run_production_batch,
    )

    # Patch FeatureStorage.get_storage_stats to avoid KeyError
    fs_module = __import__(
        "src.data_collector.indicator_pipeline.indicator_pipeline",
        fromlist=["FeatureStorage"],
    )
    fs_cls = getattr(fs_module, "FeatureStorage")
    mocker.patch.object(
        fs_cls,
        "get_storage_stats",
        return_value={"total_tickers": 2, "total_size_mb": 1.23, "base_path": "/tmp"},
    )

    res = run_production_batch()
    assert res is not None
    # consolidation key should be present when successful
    assert "consolidation" in res


def test__process_ticker_worker_date_object_conversion(mocker):
    """Convert date/datetime inputs to ISO strings before calling loader"""
    # Ensure datetime/date start/end are converted to ISO strings before loader call
    called_args = {}

    def fake_load(ticker, start, end):
        called_args["start"] = start
        called_args["end"] = end
        return pd.DataFrame({"a": [1, 2, 3]})

    mock_loader_cls = mocker.Mock()
    mock_loader = mock_loader_cls.return_value
    mock_loader.load_stock_data.side_effect = fake_load
    mocker.patch("src.feature_engineering.data_loader.StockDataLoader", mock_loader_cls)

    mock_calc = mocker.Mock()
    mock_calc.calculate_all_features.return_value = SimpleNamespace(
        data=pd.DataFrame({"f": [1, 2]}), warnings=[], quality_score=1.0
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.feature_calculator.FeatureCalculator",
        return_value=mock_calc,
    )
    mocker.patch("src.data_collector.indicator_pipeline.feature_storage.FeatureStorage")

    config_dict = {
        "start_date": datetime(2025, 1, 1),
        "end_date": date(2025, 1, 2),
        "min_data_points": 1,
        "save_to_parquet": False,
        "save_to_database": False,
    }
    res = _process_ticker_worker("T", config_dict, job_id="j")
    assert res["success"] is True
    # check that loader received ISO strings
    assert isinstance(called_args["start"], str) and called_args["start"].startswith(
        "2025-01-01"
    )
    assert isinstance(called_args["end"], str) and called_args["end"].startswith(
        "2025-01-02"
    )


def test__process_ticker_worker_save_parquet_called(mocker):
    """Call storage.save_features when save_to_parquet is True"""
    # Test that storage.save_features is called when save_to_parquet True
    mock_loader_cls = mocker.Mock()
    mock_loader = mock_loader_cls.return_value
    mock_loader.load_stock_data.return_value = pd.DataFrame({"open": [1, 2, 3]})
    mocker.patch("src.feature_engineering.data_loader.StockDataLoader", mock_loader_cls)

    mock_calc = mocker.Mock()
    feats = pd.DataFrame({"a": [1, 2, 3]})
    mock_calc.calculate_all_features.return_value = SimpleNamespace(
        data=feats, warnings=[], quality_score=0.8
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.feature_calculator.FeatureCalculator",
        return_value=mock_calc,
    )

    storage_cls = mocker.Mock()
    storage = storage_cls.return_value
    storage.save_features = mocker.Mock()
    mocker.patch(
        "src.data_collector.indicator_pipeline.feature_storage.FeatureStorage",
        storage_cls,
    )

    mocker.patch("src.database.db_utils.bulk_upsert_technical_features", return_value=3)

    config_dict = {
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "min_data_points": 1,
        "save_to_parquet": True,
        "save_to_database": False,
    }
    res = _process_ticker_worker("TICK", config_dict, job_id="job1")
    assert res["success"] is True
    storage.save_features.assert_called_once()


def test__process_ticker_worker_bulk_upsert_exception_logged(mocker):
    """Log exceptions from bulk upsert while still returning success"""
    mock_loader_cls = mocker.Mock()
    mock_loader = mock_loader_cls.return_value
    mock_loader.load_stock_data.return_value = pd.DataFrame({"open": [1, 2, 3]})
    mocker.patch("src.feature_engineering.data_loader.StockDataLoader", mock_loader_cls)

    mock_calc = mocker.Mock()
    feats = pd.DataFrame({"a": [1, 2, 3]})
    mock_calc.calculate_all_features.return_value = SimpleNamespace(
        data=feats, warnings=[], quality_score=0.8
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.feature_calculator.FeatureCalculator",
        return_value=mock_calc,
    )

    mocker.patch("src.data_collector.indicator_pipeline.feature_storage.FeatureStorage")

    # cause bulk upsert to raise
    mocker.patch(
        "src.database.db_utils.bulk_upsert_technical_features",
        side_effect=RuntimeError("boom"),
    )
    log_mock = mocker.patch(
        "src.data_collector.indicator_pipeline.indicator_pipeline.logger"
    )

    config_dict = {
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "min_data_points": 1,
        "save_to_parquet": False,
        "save_to_database": True,
    }
    res = _process_ticker_worker("TICK", config_dict, job_id="j1")
    assert res["success"] is True
    assert log_mock.error.call_count >= 1


def test_process_batch_future_exception(processor, mocker):
    """Handle exceptions raised within futures during batch processing"""

    # Arrange: make process_single_ticker raise to simulate executor future.exception path
    def raise_exc(ticker, cfg, job_id):
        """
        Always raises a RuntimeError with message "boom".
        
        Parameters:
            ticker: Identifier for the ticker (unused).
            cfg: Job configuration object or mapping (unused).
            job_id: Identifier for the job run (unused).
        
        Raises:
            RuntimeError: Always raised with the message "boom".
        """
        raise RuntimeError("boom")

    mocker.patch.object(processor, "process_single_ticker", side_effect=raise_exc)
    cfg = BatchJobConfig(
        batch_size=2, max_workers=2, use_processes=False, min_data_points=1
    )

    summary = processor.process_batch(["A", "B"], cfg)
    # both should have failed
    assert summary["failed"] >= 1


def test_run_production_batch_consolidation_failure(mocker):
    """Run production batch and capture consolidation failure path"""
    module = __import__(
        "src.data_collector.indicator_pipeline.indicator_pipeline",
        fromlist=["BatchFeatureProcessor"],
    )
    processor_cls = getattr(module, "BatchFeatureProcessor")
    proc_inst = processor_cls()
    mocker.patch.object(proc_inst, "get_available_tickers", return_value=["A"])
    mocker.patch.object(
        proc_inst,
        "process_batch",
        return_value={
            "total_tickers": 1,
            "successful": 1,
            "failed": 0,
            "results": [{"ticker": "A", "success": True}],
            "total_features": 1,
            "success_rate": 100.0,
        },
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.indicator_pipeline.BatchFeatureProcessor",
        return_value=proc_inst,
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.consolidated_storage.consolidate_existing_features",
        side_effect=RuntimeError("consolidation failed"),
    )
    fs_module = __import__(
        "src.data_collector.indicator_pipeline.indicator_pipeline",
        fromlist=["FeatureStorage"],
    )
    fs_cls = getattr(fs_module, "FeatureStorage")
    mocker.patch.object(
        fs_cls,
        "get_storage_stats",
        return_value={"total_tickers": 1, "total_size_mb": 0.1, "base_path": "/tmp"},
    )

    from src.data_collector.indicator_pipeline.indicator_pipeline import (
        run_production_batch,
    )

    res = run_production_batch()
    assert res is not None
    assert "consolidation_error" in res


def test_main_invokes_cache_clear(mocker):
    """Main should clear cleaned data cache after running production batch"""
    # Patch run_production_batch and CleanedDataCache
    mocker.patch(
        "src.data_collector.indicator_pipeline.indicator_pipeline.run_production_batch"
    )
    cache_cls = mocker.patch("src.utils.cleaned_data_cache.CleanedDataCache")
    inst = cache_cls.return_value
    inst.clear_cache = mocker.Mock()

    from src.data_collector.indicator_pipeline.indicator_pipeline import main

    main()
    inst.clear_cache.assert_called_once()
