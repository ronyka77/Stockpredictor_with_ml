"""Polars-based ML data pipeline

This module introduces a new, lazy-first Polars pipeline for preparing ML training
and prediction datasets. It is intended to be introduced alongside the existing
pandas pipeline to enable incremental migration without breaking downstream code.

Ground rules:
- Heavy ETL runs on polars.LazyFrame.
- Small .collect() operations are allowed only for:
  - feature selection boundaries that rely on sklearn/pandas
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import polars as pl

from src.data_utils.imputer import cleanup_paths as cleanup_imputer_paths
from src.data_utils.imputer import median_impute_in_subprocess
from src.utils.core.logger import get_logger
from src.utils.data.cleaned_data_cache import collect_garbage


logger = get_logger(__name__)


YEARS_TO_LOAD = (2023, 2024, 2025, 2026)

PREDICTION_CUTOFF_DATE = "2025-09-15"

EXCLUDE_COLS = {
    "ticker",
    "data_year",
    "feature_version",
    "calculation_date",
    "start_date",
    "end_date",
    "feature_categories",
    "file_path",
    "warnings",
    "quality_score",
    "record_count",
    "total_features",
    "file_size_mb",
    "days_since_last_dividend",
    "days_to_next_dividend",
    "dividend_amount",
    "dividend_yield",
    "is_ex_dividend",
}


def to_pandas(obj):
    """Convert supported objects to pandas.

    Supported inputs:
    - pandas DataFrame/Series: returned as-is
    - Polars DataFrame/Series: converted via .to_pandas()
    """

    try:
        import pandas as pd  # local import to avoid heavy dependency at import time

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj
    except Exception:
        # If pandas isn't importable for some reason, still allow polars->pandas
        # conversion via the object's own method.
        pass

    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()

    raise TypeError(f"Unsupported data type for pandas conversion: {type(obj)!r}")


def _mem(tag: str) -> None:
    logger.info(f"[MEM] {tag}")
    collect_garbage()


def _get_consolidated_storage_path() -> Path:
    """Return path to consolidated yearly parquet files.

    Mirrors ConsolidatedFeatureStorage layout without importing pandas/pyarrow.
    """

    from src.feature_engineering.config import config as fe_config

    base_path = Path(fe_config.storage.FEATURES_STORAGE_PATH)
    version = fe_config.storage.FEATURE_VERSION
    return base_path / version / "consolidated"


def _collect_int(lf: pl.LazyFrame) -> int:
    """Collect a single integer value from a one-row/one-col lazy frame."""

    return int(lf.collect().item())


def _safe_div(numer: pl.Expr, denom: pl.Expr) -> pl.Expr:
    """Safe division: returns null when denom is 0 or null."""

    return pl.when(denom.is_null() | (denom == 0)).then(None).otherwise(numer / denom)


def _schema_names(lf: pl.LazyFrame) -> set[str]:
    """Return LazyFrame column names without triggering PerformanceWarning."""

    return set(lf.collect_schema().names())


def load_all_data_lf() -> pl.LazyFrame:
    """Load consolidated yearly parquet sources as a LazyFrame.

    Replicates src.data_utils.ml_feature_loader.load_all_data() behavior:
    - scans year-partitioned parquet sources (features_YYYY.parquet)
    - adds data_year
    - left-joins ticker metadata to add ticker_id

    The join preserves original rows and allows null ticker_id.
    """

    consolidated_path = _get_consolidated_storage_path()
    if not consolidated_path.exists():
        raise FileNotFoundError(f"Consolidated path not found: {consolidated_path}")

    lfs: list[pl.LazyFrame] = []

    for year in YEARS_TO_LOAD:
        path = consolidated_path / f"features_{year}.parquet"
        if not path.exists():
            logger.warning(f"Missing consolidated parquet for year {year}: {path}")
            continue

        lf = pl.scan_parquet(str(path)).with_columns(pl.lit(year).alias("data_year"))

        lfs.append(lf)

    if not lfs:
        return pl.LazyFrame()

    lf_all = pl.concat(lfs, how="vertical_relaxed")

    # Add ticker_id via DB metadata (same behavior as pandas loader)
    try:
        from src.feature_engineering.data_loader import StockDataLoader

        meta_df = StockDataLoader().get_ticker_metadata(ticker=None)
        if getattr(meta_df, "empty", True):
            logger.error("No ticker metadata retrieved from database")
            return lf_all.with_columns(pl.lit(None).cast(pl.Int64).alias("ticker_id"))

        meta_pl = pl.DataFrame(meta_df.to_dict("list")).select(
            [pl.col("ticker"), pl.col("id").alias("ticker_id")]
        )

        lf_all = lf_all.join(meta_pl.lazy(), on="ticker", how="left")

    except Exception as exc:
        logger.warning(f"Failed to join ticker metadata: {exc}")
        lf_all = lf_all.with_columns(pl.lit(None).cast(pl.Int64).alias("ticker_id"))

    # Normalize date to Datetime (parseable)
    if "date" in _schema_names(lf_all):
        lf_all = lf_all.with_columns(
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).alias("date")
        )

    return lf_all


def add_return_target_lf(
    lf: pl.LazyFrame, *, prediction_horizon: int, filter_outliers: bool = True
) -> tuple[pl.LazyFrame, str]:
    """Add percentage return target column to the LazyFrame.

    Mirrors convert_absolute_to_percentage_returns():
    - prefer Future_Close_{h}D else first Future_High_* fallback
    - compute (future - close) / close
    - sanitize inf/NaN into null
    - optionally filter to [-0.7, 1.0] and not-null

    Returns updated LazyFrame and the new target column name.
    """

    schema_cols = _schema_names(lf)

    base_target = f"Future_Close_{prediction_horizon}D"
    if base_target in schema_cols:
        source_target = base_target
        new_target = f"Future_Return_{prediction_horizon}D"
    else:
        future_high = sorted([c for c in schema_cols if c.startswith("Future_High_")])
        if not future_high:
            raise ValueError(
                f"No future target columns found. Expected {base_target} or Future_High_* columns."
            )
        source_target = future_high[0]
        new_target = source_target.replace("Future_High_", "Future_Return_")
        logger.warning(f"Using fallback target column: {source_target}")

    if "close" not in schema_cols:
        raise ValueError("'close' column not found. Required for return calculation.")

    close = pl.col("close").cast(pl.Float64)
    future = pl.col(source_target).cast(pl.Float64)
    ret = _safe_div(future - close, close)
    ret = pl.when(ret.is_finite()).then(ret).otherwise(None)

    lf2 = lf.with_columns(ret.alias(new_target))

    if filter_outliers:
        before = _collect_int(lf2.select(pl.len()))
        lf2 = lf2.filter(
            pl.col(new_target).is_not_null()
            & (pl.col(new_target) <= 1.0)
            & (pl.col(new_target) >= -0.7)
        )
        after = _collect_int(lf2.select(pl.len()))
        dropped = before - after
        if dropped > 0:
            logger.info(
                f"Filtered {dropped} rows with invalid/outlier targets (from {before} to {after})"
            )

    return lf2, new_target


def select_feature_columns(lf: pl.LazyFrame, *, target_column: str) -> pl.LazyFrame:
    """Drop metadata and leakage columns while keeping date long enough for date features."""

    schema_cols = list(lf.collect_schema().names())

    # Drop Future_* leakage columns except the chosen return target.
    future_cols = [c for c in schema_cols if c.startswith("Future_") and c != target_column]

    # Keep date until after date features are derived.
    drop_cols = [c for c in schema_cols if (c in EXCLUDE_COLS) or (c in future_cols)]

    # Never drop the target here; callers can drop it when building X.
    drop_cols = [c for c in drop_cols if c != target_column]

    return lf.drop(drop_cols)


def add_date_features_lf(lf: pl.LazyFrame, *, date_col: str = "date") -> pl.LazyFrame:
    if date_col not in _schema_names(lf):
        logger.warning(f"Date column '{date_col}' not found. Skipping temporal features.")
        return lf

    base = pl.col(date_col)
    day_of_week0 = (base.dt.weekday() - 1).cast(pl.Int8)

    # date_int matches pandas implementation: days since 2020-01-01
    epoch = pl.datetime(2020, 1, 1)
    date_int = (base.cast(pl.Date) - epoch.cast(pl.Date)).dt.total_days().cast(pl.Int32)

    is_month_end = (base.dt.day() == base.dt.days_in_month()).cast(pl.Int8)
    is_quarter_end = (is_month_end.eq(1) & base.dt.month().is_in([3, 6, 9, 12])).cast(pl.Int8)

    return lf.with_columns(
        [
            date_int.alias("date_int"),
            base.dt.year().cast(pl.Int16).alias("year"),
            base.dt.month().cast(pl.Int8).alias("month"),
            base.dt.ordinal_day().cast(pl.Int16).alias("day_of_year"),
            base.dt.quarter().cast(pl.Int8).alias("quarter"),
            day_of_week0.alias("day_of_week"),
            is_month_end.alias("is_month_end"),
            is_quarter_end.alias("is_quarter_end"),
        ]
    )


def add_price_normalized_features_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema_cols = list(lf.collect_schema().names())

    if "close" not in schema_cols:
        logger.warning("'close' column not found. Skipping price normalization.")
        return lf

    close = pl.col("close").cast(pl.Float64)

    exprs: list[pl.Expr] = []

    # SMA ratios
    for c in schema_cols:
        if c.startswith("SMA_"):
            exprs.append(_safe_div(close, pl.col(c).cast(pl.Float64)).alias(f"{c}_Ratio"))

    # Bollinger position
    if "BB_Lower" in schema_cols and "BB_Upper" in schema_cols:
        denom = (pl.col("BB_Upper") - pl.col("BB_Lower")).cast(pl.Float64)
        numer = close - pl.col("BB_Lower").cast(pl.Float64)
        bb_pos = _safe_div(numer, denom)
        bb_pos = pl.when(bb_pos.is_finite()).then(bb_pos).otherwise(None)
        exprs.append(bb_pos.alias("BB_Position"))

    # ATR normalized
    if "ATR" in schema_cols:
        exprs.append(_safe_div(close, pl.col("ATR").cast(pl.Float64)).alias("Price_ATR_Ratio"))

    # Return/volume efficiency
    if "volume" in schema_cols and "Return_1D" in schema_cols:
        denom = pl.col("volume").cast(pl.Float64) + 1e-8
        numer = pl.col("Return_1D").cast(pl.Float64).abs()
        exprs.append((numer / denom).alias("Return_Volume_Efficiency"))

    # Ichimoku ratios (skip boolean-ish derived columns)
    for c in schema_cols:
        if not c.startswith("Ichimoku_"):
            continue
        if c.endswith(("_Above_Kijun", "_Above_Cloud", "_Below_Cloud", "_Green", "_Thickness")):
            continue
        exprs.append(_safe_div(close, pl.col(c).cast(pl.Float64)).alias(f"{c}_Ratio"))

    # Close/Open ratio
    if "open" in schema_cols:
        exprs.append(_safe_div(close, pl.col("open").cast(pl.Float64)).alias("Close_Open_Ratio"))

    return lf.with_columns(exprs) if exprs else lf


def add_prediction_bounds_features_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema_cols = _schema_names(lf)
    exprs: list[pl.Expr] = []

    if "ATR_Percent" in schema_cols:
        atrp = pl.col("ATR_Percent").cast(pl.Float64)
        exprs.append(atrp.alias("Expected_Daily_Move"))
        exprs.append((atrp * (10.0**0.5)).alias("Expected_10D_Move"))

    if "Vol_Regime_High" in schema_cols and "Vol_Regime_Low" in schema_cols:
        exprs.append(
            (
                pl.col("Vol_Regime_High").cast(pl.Float64) * 2.0
                + pl.col("Vol_Regime_Low").cast(pl.Float64) * 0.5
            ).alias("Vol_Regime_Context")
        )

    if "RSI_14" in schema_cols:
        exprs.append(
            ((pl.col("RSI_14").cast(pl.Float64) - 50.0).abs() / 50.0).alias(
                "RSI_Mean_Reversion_Pressure"
            )
        )

    return lf.with_columns(exprs) if exprs else lf


def sanitize_numeric_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Replace inf/-inf/NaN with null and cast numeric columns to Float32."""

    exprs: list[pl.Expr] = []

    for name, dtype in lf.collect_schema().items():
        if name in {"ticker_id"}:
            continue

        if dtype.is_numeric():
            col = pl.col(name)
            # float columns: NaN -> null
            col = col.fill_nan(None)
            # any numeric: non-finite -> null
            col = pl.when(col.is_finite()).then(col).otherwise(None)
            # cast to float32 to reduce memory pressure
            col = col.cast(pl.Float32)
            exprs.append(col.alias(name))

    return lf.with_columns(exprs) if exprs else lf


def split_train_test_lf(
    lf: pl.LazyFrame, *, split_date: str, date_col: str = "date"
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    if date_col not in _schema_names(lf):
        raise ValueError(f"'{date_col}' column not found. Cannot split.")

    split_dt = pl.lit(split_date).str.strptime(pl.Date, strict=False)
    date_expr = pl.col(date_col).cast(pl.Date)

    train = lf.filter(date_expr < split_dt)
    test = lf.filter(date_expr >= split_dt)
    return train, test


def filter_weekdays_lf(
    lf: pl.LazyFrame, *, date_col: str = "date", weekdays: tuple[int, ...] = (0, 4)
) -> pl.LazyFrame:
    if date_col not in _schema_names(lf):
        raise ValueError(f"'{date_col}' column not found. Cannot filter weekdays.")

    dow0 = (pl.col(date_col).dt.weekday() - 1).cast(pl.Int8)
    return lf.filter(dow0.is_in(list(weekdays)))


def _prepare_base_ml_lf(
    *, prediction_horizon: int, filter_outliers: bool
) -> tuple[pl.LazyFrame, str]:
    """Shared base preparation for Polars ML pipelines.

    Builds a LazyFrame with:
    - consolidated sources loaded
    - percentage return target added
    - metadata/leakage columns dropped
    - feature engineering applied
    - numeric sanitization applied

    Callers can then apply split/filtering and collect.
    """

    lf = load_all_data_lf()
    if len(lf.collect_schema()) == 0:
        raise ValueError("No data loaded. Check data availability.")

    if "date" not in _schema_names(lf):
        raise ValueError("'date' column not found in data. Cannot proceed.")

    lf, target_column = add_return_target_lf(
        lf, prediction_horizon=prediction_horizon, filter_outliers=filter_outliers
    )
    lf = select_feature_columns(lf, target_column=target_column)
    if filter_outliers:
        lf = lf.filter(pl.col(target_column).is_not_null())

    lf = add_price_normalized_features_lf(lf)
    lf = add_prediction_bounds_features_lf(lf)
    lf = add_date_features_lf(lf, date_col="date")
    lf = sanitize_numeric_lf(lf)

    return lf, target_column


def prepare_ml_data_for_training_with_cleaning_polars(
    *, prediction_horizon: int = 10, split_date: str = "2025-05-01", clean_features: bool = True
) -> dict:
    """Polars-native ML training data preparation (Phase 1).

    Returns a dict with keys aligned to the pandas pipeline, but with Polars objects:
    - x_train/x_test: pl.DataFrame
    - y_train/y_test: pl.Series

    Weekday filtering is applied only to the test set (Mon/Fri: 0,4).
    """

    logger.info(
        f"[START] Polars ML data prep (horizon: {prediction_horizon}d, split: {split_date})"
    )

    lf = load_all_data_lf()
    if len(lf.collect_schema()) == 0:
        raise ValueError("No data loaded. Check data availability.")

    if "date" not in _schema_names(lf):
        raise ValueError("'date' column not found in data. Cannot perform date-based split.")

    lf, target_column = add_return_target_lf(
        lf, prediction_horizon=prediction_horizon, filter_outliers=True
    )
    # Drop leakage + metadata columns (keep date for date feature creation)
    lf = select_feature_columns(lf, target_column=target_column)

    # Keep only rows with a valid target
    lf = lf.filter(pl.col(target_column).is_not_null())

    # Feature engineering (lazy)
    lf = add_price_normalized_features_lf(lf)
    lf = add_prediction_bounds_features_lf(lf)
    lf = add_date_features_lf(lf, date_col="date")

    # Cleaning (lazy-first)
    lf = sanitize_numeric_lf(lf)

    # Split
    train_lf, test_lf = split_train_test_lf(lf, split_date=split_date, date_col="date")

    # Weekday filter applies ONLY to test set
    test_lf = filter_weekdays_lf(test_lf, date_col="date", weekdays=(0, 4))

    # Median imputation (per ticker_id) uses TRAIN statistics only.
    # This avoids cross-ticker blending and prevents test leakage.
    num_cols = [
        c
        for c, t in train_lf.collect_schema().items()
        if t.is_numeric() and c not in {"ticker_id"} and c != target_column
    ]

    cleanup: list[Path] = []
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="polars_median_impute_"))
        train_lf, test_lf, cleanup = median_impute_in_subprocess(
            train_lf=train_lf,
            test_lf=test_lf,
            num_cols=num_cols,
            work_dir=work_dir,
            group_key="ticker_id",
        )
        _mem("after median_impute subprocess")

        # Drop date column after date features exist (pandas parity)
        for_drop = [c for c in ("date",) if c in _schema_names(train_lf)]
        if for_drop:
            train_lf = train_lf.drop(for_drop)
            test_lf = test_lf.drop(for_drop)

        # Hybrid feature selection boundary
        removed_features: dict | None = None
        if clean_features:
            # Collect just before sklearn selection
            train_pd = train_lf.collect().to_pandas()
            x_train_pd = train_pd.drop(columns=[target_column])

            features_to_keep = list(x_train_pd.columns)

            train_lf = train_lf.select([*features_to_keep, target_column])
            test_lf = test_lf.select([*features_to_keep, target_column])

        # Collect at the pipeline boundary (returns eager Polars objects)
        train_df = train_lf.collect()
        test_df = test_lf.collect()
        if "date_int" in test_df.columns:
            test_df = test_df.sort("date_int", descending=True)
        # test_df.head(50000).write_excel("debug_test_data.xlsx")  # Debug output

    finally:
        if cleanup:
            cleanup_imputer_paths(cleanup)

    if train_df.height == 0:
        raise ValueError(f"No training data found before {split_date}. Check your date range.")
    if test_df.height == 0:
        raise ValueError(f"No test data found after {split_date}. Check your date range.")

    x_train = train_df.drop(target_column)
    y_train = train_df[target_column]
    x_test = test_df.drop(target_column)
    y_test = test_df[target_column]

    # Date range reporting
    train_date_range = f"2023-01-01 to {split_date}"
    test_date_range = f"{split_date} onward"

    result = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_column": target_column,
        "removed_features": removed_features or {},
        "feature_count": x_train.width,
        "train_samples": x_train.height,
        "test_samples": x_test.height,
        "prediction_horizon": prediction_horizon,
        "split_date": split_date,
        "train_date_range": train_date_range,
        "test_date_range": test_date_range,
    }

    logger.info(
        f"[END] Polars data prep completed: {result['train_samples']} train, {result['test_samples']} test, {result['feature_count']} features"
    )

    _mem("end")

    return result


def prepare_ml_data_for_prediction_with_cleaning_polars(
    *, prediction_horizon: int = 10, days_back: int = 30
) -> dict:
    """Polars-native ML prediction data preparation (Phase 1).

    Mirrors prepare_ml_data_for_prediction_with_cleaning() behavior where practical:
    - percentage return targets with outlier filtering disabled
    - weekday filtering (Mon/Fri)
    - cutoff date selection (see PREDICTION_CUTOFF_DATE)
    - optional days_back filter using date_int

    Returns dict keys aligned to pandas prediction pipeline, but with Polars objects:
    - x_test: pl.DataFrame
    - y_test: pl.Series
    """

    logger.info(
        f"[START] Polars prediction data prep (horizon: {prediction_horizon}d, days_back: {days_back})"
    )

    lf, target_column = _prepare_base_ml_lf(
        prediction_horizon=prediction_horizon, filter_outliers=False
    )

    # Pandas prediction path filters to Mondays/Fridays early.
    lf = filter_weekdays_lf(lf, date_col="date", weekdays=(0, 4))

    # Pandas prediction path uses a fixed cutoff date to form the prediction set.
    cutoff_dt = pl.lit(PREDICTION_CUTOFF_DATE).str.strptime(pl.Date, strict=False)
    lf = lf.filter(pl.col("date").cast(pl.Date) >= cutoff_dt)

    # Optional: narrow to recent rows based on date_int.
    if days_back and "date_int" in _schema_names(lf):
        try:
            max_date_int = _collect_int(lf.select(pl.col("date_int").max()))
            lf = lf.filter(pl.col("date_int") >= (max_date_int - int(days_back)))
        except Exception as exc:
            logger.warning(f"days_back filter skipped: {exc}")

    # Median imputation in a spawned subprocess to release memory back to the OS on Windows.
    # We treat the prediction set as both "train" and "test" here (no split), which preserves
    # the same median-impute semantics without extra in-process allocations.
    num_cols = [
        c
        for c, t in lf.collect_schema().items()
        if t.is_numeric() and c not in {"ticker_id", target_column}
    ]

    cleanup: list[Path] = []
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="polars_median_impute_pred_"))
        _train_lf, imputed_lf, cleanup = median_impute_in_subprocess(
            train_lf=lf, test_lf=lf, num_cols=num_cols, work_dir=work_dir, group_key="ticker_id"
        )
        _mem("after prediction median_impute subprocess")
        df = imputed_lf.collect()
    finally:
        if cleanup:
            cleanup_imputer_paths(cleanup)

    if df.height == 0:
        raise ValueError(
            f"No prediction data found on/after {PREDICTION_CUTOFF_DATE}. Check your date range."
        )

    # Date range reporting (best-effort)
    test_date_range = f"{PREDICTION_CUTOFF_DATE} onward"
    if "date" in df.columns:
        dmin, dmax = df.select(
            pl.col("date").min().alias("date_min"), pl.col("date").max().alias("date_max")
        ).row(0)
        if dmin is not None and dmax is not None:
            test_date_range = f"{dmin:%Y-%m-%d} to {dmax:%Y-%m-%d}"

    # Drop date column after date features exist (pandas parity)
    if "date" in df.columns:
        df = df.drop("date")

    x_test = df.drop(target_column)
    y_test = df[target_column]

    result = {
        "x_test": x_test,
        "y_test": y_test,
        "target_column": target_column,
        "test_date_range": test_date_range,
        "feature_count": x_test.width,
        "test_samples": x_test.height,
        "prediction_horizon": prediction_horizon,
        "days_back": days_back,
    }

    logger.info(
        f"[END] Polars prediction prep completed: {result['test_samples']} samples, {result['feature_count']} features"
    )
    _mem("end prediction")
    return result


if __name__ == "__main__":
    prepare_ml_data_for_training_with_cleaning_polars(prediction_horizon=20)
